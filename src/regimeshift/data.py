"""NASA C-MAPSS FD002/FD004 (six operating conditions) as a regime-shift anomaly benchmark.

Per cycle: 3 operating settings + 21 sensors. Normal = RUL >= 100 cycles; degraded (anomaly) = RUL <= 20.
Cycles in between are excluded from evaluation (ambiguous health).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

DIR = Path(os.environ.get("INDUSTRIAL_TS_DATA", "/workspace/industrial-ts-data")) / "CMAPSS"
COLS = ["unit", "cycle", "s1", "s2", "s3"] + [f"m{i}" for i in range(1, 22)]
SETTINGS = ["s1", "s2", "s3"]; SENSORS = [f"m{i}" for i in range(1, 22)]
NORMAL_RUL, ANOM_RUL = 100, 20


def load(subset: str = "FD002") -> pd.DataFrame:
    df = pd.read_csv(DIR / f"train_{subset}.txt", sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    df["rul"] = df.groupby("unit").cycle.transform("max") - df.cycle
    df["label"] = np.where(df.rul >= NORMAL_RUL, 0, np.where(df.rul <= ANOM_RUL, 1, -1))
    return df


def discover_regimes(settings: np.ndarray, k: int, seed: int = 0) -> KMeans:
    """Regime discovery: k-means on standardized operating settings (no labels)."""
    mu = settings.mean(0); sd = np.maximum(settings.std(0), 1e-2 * (np.abs(mu) + 1))  # floor: a setting may be constant in seen regimes
    km = KMeans(k, n_init=10, random_state=seed).fit((settings - mu) / sd)
    km.mu_, km.sd_ = mu, sd
    return km


def assign(km: KMeans, settings: np.ndarray) -> np.ndarray:
    return km.predict((settings - km.mu_) / km.sd_)


def split_units(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0):
    u = df.unit.unique(); rng = np.random.default_rng(seed); rng.shuffle(u)
    nt = int(len(u) * test_frac); return set(u[nt:]), set(u[:nt])
