"""RegimeShift-Invariant experiments E1-E5 on C-MAPSS FD002/FD004 (leave-one-regime-out)."""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from regimeshift import data, models

cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
K, LAT, INV, CONF, FA, EP, SEEDS = (cfg[k] for k in ["n_regimes", "latent_dim", "invariance", "confusion", "false_alarm_target", "epochs", "seeds"])
RES = ROOT / "experiments/results"; RES.mkdir(parents=True, exist_ok=True)
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
SUBSETS = [a for a in sys.argv[1:] if a.startswith("FD")] or ["FD002", "FD004"]
EXPS = [a for a in sys.argv[1:] if a.startswith("E")] or ["E1", "E4", "E5"]


def positives(df_tr, R):
    """Positive partner: same unit, |dcycle|<=3, preferably a different regime."""
    pos = np.full(len(df_tr), -1); u = df_tr.unit.values; c = df_tr.cycle.values
    rng = np.random.default_rng(0)
    for unit in np.unique(u):
        idx = np.where(u == unit)[0]
        for a, i in enumerate(idx):
            cand = idx[max(0, a - 3):a + 4]; cand = cand[cand != i]
            diff = cand[R[cand] != R[i]]
            pool = diff if len(diff) else cand
            if len(pool): pos[i] = rng.choice(pool)
    return pos


SD_FLOOR = None  # set per fold: 1% of global training std, guards sensors that are constant within a regime
def zfit(x):
    sd = x.std(0)
    if SD_FLOOR is not None: sd = np.maximum(sd, SD_FLOOR)
    return x.mean(0), sd + 1e-6
CLIP = 10.0  # z-scores clipped to +-10 for every method


def run_fold(df, sens, k_regimes, holdout_true, seed, methods):
    """Train on normal cycles of training units in seen regimes; evaluate on test units."""
    tr_u, te_u = data.split_units(df, 0.3, seed)
    km_true = data.discover_regimes(df[data.SETTINGS].values, 6, 0)          # reference 6 conditions to define the holdout
    df = df.assign(true_reg=data.assign(km_true, df[data.SETTINGS].values))
    trm = df.unit.isin(tr_u) & (df.label == 0) & (df.true_reg != holdout_true)
    dtr = df[trm].reset_index(drop=True); dte = df[df.unit.isin(te_u) & (df.label >= 0)].reset_index(drop=True)
    km = data.discover_regimes(dtr[data.SETTINGS].values, k_regimes, seed)   # regimes discovered on training data only
    Rtr = data.assign(km, dtr[data.SETTINGS].values); Rte = data.assign(km, dte[data.SETTINGS].values)
    Xtr, Xte = dtr[sens].values, dte[sens].values
    Ctr = np.clip((dtr[data.SETTINGS].values - km.mu_) / km.sd_, -CLIP, CLIP); Cte = np.clip((dte[data.SETTINGS].values - km.mu_) / km.sd_, -CLIP, CLIP)
    global SD_FLOOR; SD_FLOOR = None
    mu, sd = zfit(Xtr); SD_FLOOR = 0.01 * sd
    Gtr, Gte = np.clip((Xtr - mu) / sd, -CLIP, CLIP), np.clip((Xte - mu) / sd, -CLIP, CLIP)
    Ptr = np.zeros_like(Xtr); Pte = np.zeros_like(Xte); Tte = np.zeros_like(Xte)
    stats = {r: zfit(Xtr[Rtr == r]) for r in np.unique(Rtr)}
    for r, (m_, s_) in stats.items(): Ptr[Rtr == r] = (Xtr[Rtr == r] - m_) / s_
    for r in np.unique(Rte):
        m_, s_ = stats[r] if r in stats else stats[min(stats)]
        Pte[Rte == r] = (Xte[Rte == r] - m_) / s_
    # transductive variant: target-regime statistics from UNLABELED test cycles of the held-out condition
    unseen = dte.true_reg.values == holdout_true
    Tte[:] = Pte; mt, st = zfit(Xte[unseen]); Tte[unseen] = (Xte[unseen] - mt) / st
    Ptr, Pte, Tte = (np.clip(a, -CLIP, CLIP) for a in (Ptr, Pte, Tte))
    pos = positives(dtr, Rtr)
    out = {}
    def ae(name, Xa, Xb, Ca, Cb, **kw):
        m = models.train(Xa, Ca, Rtr, pos, seed, latent=LAT, epochs=EP, n_regimes=k_regimes, **kw)
        sa, _ = models.score_embed(m, Xa, Ca); sb, zb = models.score_embed(m, Xb, Cb); out[name] = (sa, sb, zb)
    if "Global norm + PCA-SPE" in methods:
        p = PCA(0.95).fit(Gtr); f = lambda g: ((g - p.inverse_transform(p.transform(g))) ** 2).mean(1)
        out["Global norm + PCA-SPE"] = (f(Gtr), f(Gte), p.transform(Gte))
    if "Global norm + AE" in methods: ae("Global norm + AE", Gtr, Gte, Ctr, Cte, film=False)
    if "Per-regime norm + AE" in methods: ae("Per-regime norm + AE", Ptr, Pte, Ctr, Cte, film=False)
    if "Per-regime norm + AE (transductive target stats)" in methods:
        m = models.train(Ptr, Ctr, Rtr, pos, seed, film=False, latent=LAT, epochs=EP, n_regimes=k_regimes)
        sa, _ = models.score_embed(m, Ptr, Ctr); sb, zb = models.score_embed(m, Tte, Cte); out["Per-regime norm + AE (transductive target stats)"] = (sa, sb, zb)
    if "DANN AE (GRL only)" in methods: ae("DANN AE (GRL only)", Gtr, Gte, Ctr, Cte, film=False, confusion=CONF)
    if "FiLM-Inv (ours)" in methods: ae("FiLM-Inv (ours)", Gtr, Gte, Ctr, Cte, film=True, invariance=INV, confusion=CONF)
    if "FiLM only" in methods: ae("FiLM only", Gtr, Gte, Ctr, Cte, film=True)
    if "FiLM + contrastive (no GRL)" in methods: ae("FiLM + contrastive (no GRL)", Gtr, Gte, Ctr, Cte, film=True, invariance=INV)
    if "FiLM + GRL (no contrastive)" in methods: ae("FiLM + GRL (no contrastive)", Gtr, Gte, Ctr, Cte, film=True, confusion=CONF)
    if "No FiLM (contrastive + GRL)" in methods: ae("No FiLM (contrastive + GRL)", Gtr, Gte, Ctr, Cte, film=False, invariance=INV, confusion=CONF)
    rows = []
    y = dte.label.values
    for name, (sa, sb, zb) in out.items():
        thr = np.quantile(sa, 1 - FA)
        r = dict(method=name, holdout_regime=holdout_true, seed=seed, k_regimes=k_regimes,
                 auroc_unseen=roc_auc_score(y[unseen], sb[unseen]), auroc_seen=roc_auc_score(y[~unseen], sb[~unseen]),
                 far_seen=float((sb[~unseen & (y == 0)] > thr).mean()), far_unseen=float((sb[unseen & (y == 0)] > thr).mean()),
                 tpr_unseen=float((sb[unseen & (y == 1)] > thr).mean()))
        # E2: false alarms on the first normal cycle after a switch from a seen into the unseen regime
        prev_seen = np.r_[False, (~unseen[:-1]) & (dte.unit.values[1:] == dte.unit.values[:-1])]
        sw = unseen & prev_seen & (y == 0); r["far_after_switch"] = float((sb[sw] > thr).mean()) if sw.any() else np.nan
        # E3: how much regime information is left in the representation (6 true conditions, lower = more invariant)
        sub = np.random.default_rng(seed).choice(len(zb), min(4000, len(zb)), replace=False)
        r["regime_probe_acc"] = cross_val_score(LogisticRegression(max_iter=500), zb[sub], dte.true_reg.values[sub], cv=3).mean()
        rows.append(r)
    return rows


MAIN = ["Global norm + PCA-SPE", "Global norm + AE", "Per-regime norm + AE", "Per-regime norm + AE (transductive target stats)", "DANN AE (GRL only)", "FiLM-Inv (ours)"]
ABL = ["FiLM-Inv (ours)", "FiLM only", "FiLM + contrastive (no GRL)", "FiLM + GRL (no contrastive)", "No FiLM (contrastive + GRL)"]
for sub in SUBSETS:
    df = data.load(sub)
    sens = [s for s in data.SENSORS if df[s].std() > 1e-4]
    if "E1" in EXPS:
        rows = []
        for s in SEEDS:
            for h in range(6):
                rows += run_fold(df, sens, K, h, s, MAIN); log(sub, "E1", s, h)
        pd.DataFrame(rows).to_csv(RES / f"e1_e2_e3_{sub}_raw.csv", index=False)
    if "E4" in EXPS:
        rows = []
        for s in SEEDS:
            for kk in [2, 3, 4, 6, 8, 12]:
                for h in [0, 3]:
                    rows += run_fold(df, sens, kk, h, s, ["Per-regime norm + AE", "FiLM-Inv (ours)"]); log(sub, "E4", s, kk, h)
        pd.DataFrame(rows).to_csv(RES / f"e4_{sub}_raw.csv", index=False)
    if "E5" in EXPS:
        rows = []
        for s in SEEDS:
            for h in range(6):
                rows += run_fold(df, sens, K, h, s, ABL[1:]); log(sub, "E5", s, h)
        pd.DataFrame(rows).to_csv(RES / f"e5_{sub}_raw.csv", index=False)
log("done")
