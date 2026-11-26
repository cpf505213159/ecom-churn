"""模型核心逻辑的单元测试。

重点测三件容易写错、错了又不报错的事：
1. AUC 在完全可分/完全随机时的边界值
2. 按承接量切名单的召回与 lift 定义
3. 泄漏特征确实会把 AUC 抬高（如果测不出来，说明生成器或特征白名单有问题）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gen import build  # noqa: E402
from src.model import (  # noqa: E402
    FEATURES,
    LEAKY,
    at_capacity,
    auc,
    cost_model,
    fit_logreg,
    ks,
    predict_proba,
)


def test_auc_boundaries() -> None:
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert abs(auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-12


def test_auc_handles_ties() -> None:
    # 有并列时 AUC 应为 0.5（完全无信息）
    y = np.array([1, 0, 1, 0])
    s = np.array([0.3, 0.3, 0.3, 0.3])
    assert abs(auc(y, s) - 0.5) < 1e-12


def test_ks_range() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = y * 0.6 + rng.normal(0, 1, 2000)
    v = ks(y, s)
    assert 0 <= v <= 1
    assert v > 0.2  # 有明显信号时 KS 应该不小


def test_capacity_selection() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    r = at_capacity(y, s, 0.2)
    assert r["n_selected"] == 2
    assert r["recall"] == 1.0
    assert abs(r["churn_rate_in_selected"] - 1.0) < 1e-12
    assert r["lift"] == 5.0  # 大盘 0.2，名单 1.0 → 5 倍


def test_leaky_features_inflate_auc() -> None:
    df, _ = build(n_users=20000)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.7)
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]

    def score(feats):
        Xtr, Xte = tr[feats].to_numpy(float), te[feats].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1
        w = fit_logreg((Xtr - mu) / sd, tr["churn"].to_numpy(float), l2=2.0)
        return predict_proba((Xte - mu) / sd, w)

    y = te["churn"].to_numpy()
    clean = auc(y, score(FEATURES))
    leaky = auc(y, score(FEATURES + LEAKY))
    assert leaky > clean, (clean, leaky)


def test_cost_model() -> None:
    c = cost_model(1_000_000, 0.1, cost_per_touch=0.5)
    assert c["touched"] == 100_000
    assert c["cost"] == 50_000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"✓ {name}")
