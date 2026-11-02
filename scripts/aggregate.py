"""Aggregate raw CSVs -> summaries (mean/std over seeds, averaged over held-out regimes), figures, reports/RESULTS.md."""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "experiments/results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
M = ["auroc_unseen", "auroc_seen", "far_seen", "far_unseen", "tpr_unseen", "far_after_switch", "regime_probe_acc"]
out = ["# RegimeShift — Results", "", "Aggregated from `experiments/results/*_raw.csv`. Each seed's value is first averaged over held-out regimes (leave-one-regime-out); mean ± std is then taken over seeds 0,1,2.", ""]
def summ(d, keys):
    per_seed = d.groupby(keys + ["seed"], sort=False)[M].mean().reset_index()
    g = per_seed.groupby(keys, sort=False)[M].agg(["mean", "std"]).round(3); g.columns = [f"{a}_{b}" for a, b in g.columns]; return g.reset_index()
for sub in ["FD002", "FD004"]:
    f = RES / f"e1_e2_e3_{sub}_raw.csv"
    if not f.exists(): continue
    d1 = pd.read_csv(f); s = summ(d1, ["method"]); s.to_csv(RES / f"e1_e2_e3_{sub}_summary.csv", index=False)
    out += [f"## {sub} — E1 unseen regime / E2 false alarms after switch / E3 regime probe", "", s.to_markdown(index=False), ""]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].barh(s.method, s.auroc_unseen_mean, xerr=s.auroc_unseen_std, color="C0", label="unseen"); ax[0].scatter(s.auroc_seen_mean, s.method, c="k", marker="|", s=200, label="seen"); ax[0].set_xlabel("AUROC"); ax[0].legend(fontsize=7)
    ax[1].barh(s.method, s.far_unseen_mean, xerr=s.far_unseen_std, color="C3"); ax[1].axvline(0.01, ls="--", c="k"); ax[1].set_xlabel("FAR in unseen regime (target 0.01)"); ax[1].set_yticklabels([])
    ax[2].scatter(s.regime_probe_acc_mean, s.auroc_unseen_mean)
    for _, r in s.iterrows(): ax[2].annotate(r.method[:22], (r.regime_probe_acc_mean, r.auroc_unseen_mean), fontsize=6)
    ax[2].set_xlabel("regime probe accuracy (lower = more invariant)"); ax[2].set_ylabel("unseen AUROC")
    plt.suptitle(sub); plt.tight_layout(); plt.savefig(FIG / f"e1_e3_{sub}.png", dpi=150); plt.close()
    f = RES / f"e4_{sub}_raw.csv"
    if f.exists():
        d = pd.read_csv(f); s4 = summ(d, ["method", "k_regimes"]); s4.to_csv(RES / f"e4_{sub}_summary.csv", index=False)
        out += [f"## {sub} — E4 misspecified number of regimes (holdouts 0,3; true = 6)", "", s4[["method", "k_regimes", "auroc_unseen_mean", "auroc_unseen_std", "far_unseen_mean", "far_unseen_std"]].to_markdown(index=False), ""]
        plt.figure(figsize=(5, 3.5))
        for m, g in s4.groupby("method"): plt.errorbar(g.k_regimes, g.auroc_unseen_mean, g.auroc_unseen_std, marker="o", capsize=3, label=m)
        plt.axvline(6, ls=":", c="k"); plt.xlabel("K used by KMeans"); plt.ylabel("unseen AUROC"); plt.legend(fontsize=7); plt.title(sub); plt.tight_layout(); plt.savefig(FIG / f"e4_{sub}.png", dpi=150); plt.close()
    f = RES / f"e5_{sub}_raw.csv"
    if f.exists():
        d = pd.concat([d1[d1.method == "FiLM-Inv (ours)"], pd.read_csv(f)]); s5 = summ(d, ["method"]); s5.to_csv(RES / f"e5_{sub}_summary.csv", index=False)
        out += [f"## {sub} — E5 ablations", "", s5.to_markdown(index=False), ""]
(ROOT / "reports").mkdir(exist_ok=True); (ROOT / "reports/RESULTS.md").write_text("\n".join(out)); print("\n".join(out))
