"""Autoencoder family: plain, DANN-style, and FiLM-conditioned with invariance losses."""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


class FiLM(nn.Module):
    """h -> gamma(c) * h + beta(c); condition c = operating settings."""
    def __init__(self, cdim, h):
        super().__init__(); self.net = nn.Sequential(nn.Linear(cdim, 32), nn.ReLU(), nn.Linear(32, 2 * h))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, h, c):
        g, b = self.net(c).chunk(2, -1); return (1 + g) * h + b


class RegimeAE(nn.Module):
    def __init__(self, d, cdim, latent=8, hidden=64, film=True, n_regimes=6):
        super().__init__()
        self.film = film
        self.e1, self.e2 = nn.Linear(d, hidden), nn.Linear(hidden, latent)
        self.d1, self.d2 = nn.Linear(latent, hidden), nn.Linear(hidden, d)
        if film: self.fe, self.fd = FiLM(cdim, hidden), FiLM(cdim, hidden)
        self.cls = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, n_regimes))

    def encode(self, x, c):
        h = self.e1(x); h = self.fe(h, c) if self.film else h
        return self.e2(F.relu(h))

    def decode(self, z, c):
        h = self.d1(z); h = self.fd(h, c) if self.film else h
        return self.d2(F.relu(h))

    def forward(self, x, c):
        z = self.encode(x, c); return z, self.decode(z, c)


def info_nce(za, zp, tau=0.2):
    za, zp = F.normalize(za, dim=-1), F.normalize(zp, dim=-1)
    logits = za @ zp.T / tau; return F.cross_entropy(logits, torch.arange(len(za)))


def train(X, C, R, pos, seed, film=True, invariance=0.0, confusion=0.0, latent=8, epochs=20, n_regimes=6):
    """X sensors, C settings, R regime ids, pos: index of a positive partner (same unit, nearby cycle) or -1."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = RegimeAE(X.shape[1], C.shape[1], latent, film=film, n_regimes=max(n_regimes, int(R.max()) + 1))
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt, Ct, Rt, Pt = map(torch.from_numpy, (X.astype(np.float32), C.astype(np.float32), R.astype(np.int64), pos.astype(np.int64)))
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n); lam = confusion * (2 / (1 + np.exp(-10 * ep / epochs)) - 1)
        for b in range(0, n, 256):
            i = perm[b:b + 256]; z, xh = m(Xt[i], Ct[i]); loss = F.mse_loss(xh, Xt[i])
            if confusion > 0:
                loss = loss + F.cross_entropy(m.cls(GradReverse.apply(z, lam)), Rt[i])
            if invariance > 0:
                j = Pt[i]; ok = j >= 0
                if ok.sum() > 4:
                    zp = m.encode(Xt[j[ok]], Ct[j[ok]]); loss = loss + invariance * info_nce(z[ok], zp)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def score_embed(m, X, C):
    Xt, Ct = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(C.astype(np.float32))
    z, xh = m(Xt, Ct); return ((xh - Xt) ** 2).mean(1).numpy(), z.numpy()
