import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch
from regimeshift import data, models


def test_regime_discovery_recovers_clusters():
    rng = np.random.default_rng(0); centers = np.array([[0, 0, 100], [20, .7, 100], [42, .84, 60]])
    S = np.vstack([c + rng.normal(0, .01, (100, 3)) for c in centers]); km = data.discover_regimes(S, 3)
    r = data.assign(km, S); assert len(set(r[:100])) == 1 and len({r[0], r[100], r[200]}) == 3


def test_constant_setting_is_floored():
    S = np.c_[np.random.default_rng(0).normal(size=(50, 2)), np.full(50, 100.0)]
    km = data.discover_regimes(S, 2); assert np.isfinite(km.sd_).all() and km.sd_[2] > 0.5


def test_grad_reverse_flips_gradient():
    x = torch.ones(3, requires_grad=True); models.GradReverse.apply(x, 2.0).sum().backward()
    assert torch.allclose(x.grad, -2 * torch.ones(3))


def test_film_identity_at_init():
    f = models.FiLM(3, 5); h = torch.randn(4, 5); assert torch.allclose(f(h, torch.randn(4, 3)), h)


def test_info_nce_lower_for_matched_pairs():
    z = torch.randn(16, 8); assert models.info_nce(z, z) < models.info_nce(z, torch.randn(16, 8))


def test_train_and_score_shapes():
    rng = np.random.default_rng(0); X = rng.normal(size=(300, 6)); C = rng.normal(size=(300, 3)); R = rng.integers(0, 3, 300)
    pos = np.r_[np.arange(1, 300), -1]
    m = models.train(X, C, R, pos, 0, film=True, invariance=.5, confusion=.3, epochs=2, n_regimes=3)
    s, z = models.score_embed(m, X, C); assert s.shape == (300,) and z.shape == (300, 8)


def test_labels_definition():
    import pandas as pd
    if not (data.DIR / "train_FD002.txt").exists(): return
    df = data.load("FD002"); assert set(df.label.unique()) == {-1, 0, 1}
    assert (df[df.label == 1].rul <= data.ANOM_RUL).all()
