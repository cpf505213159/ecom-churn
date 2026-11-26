"""流失模型的最小实现：L2 逻辑回归 + AUC/KS + 按承接量定阈值。

为什么不用 sklearn
------------------
公司环境里通常能装，但这个包要保证「clone 下来没有任何依赖也能跑」，
而且逻辑回归的牛顿法只有十几行，自己写反而能把「为什么用时间外验证」
「阈值为什么按业务承接量倒推」这些判断讲清楚。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 特征白名单：明确排除券相关字段（运营干预的结果，会造成数据泄漏）
FEATURES = [
    "tenure_days", "recency_days", "freq_30d", "orders_90d", "gmv_90d",
    "browse_30d", "search_30d", "cart_30d", "tickets_30d",
    "return_rate_90d", "cat_diversity", "is_new",
]
LEAKY = ["coupon_received_30d", "coupon_used_30d"]


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 30) -> np.ndarray:
    """牛顿法（IRLS）求解 L2 正则逻辑回归，返回含截距的系数。"""
    n, k = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(k + 1)
    reg = np.eye(k + 1) * l2
    reg[0, 0] = 0.0  # 截距不正则

    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        grad = Xb.T @ (p - y) + reg @ w
        W = p * (1 - p) + 1e-9
        H = Xb.T @ (Xb * W[:, None]) + reg
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-(Xb @ w)))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """基于秩的 AUC，并列分数取平均秩（不依赖 scipy）。"""
    y = np.asarray(y)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)

    # 处理并列：同一分数取平均秩
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ks(y: np.ndarray, score: np.ndarray) -> float:
    """KS = 正负样本累计分布的最大间距，风控/流失场景常用。"""
    df = pd.DataFrame({"y": y, "s": score}).sort_values("s", ascending=False)
    pos = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    neg = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def at_capacity(y: np.ndarray, score: np.ndarray, capacity_frac: float) -> dict:
    """按「运营能承接的人数」切阈值，而不是按 F1 最优点。

    capacity_frac 是全量用户里能被触达的比例。真实项目里它是运营给的硬约束
    （短信条数、客服人力、券预算），不是统计参数。
    """
    n = len(y)
    k = max(int(n * capacity_frac), 1)
    order = np.argsort(-score)[:k]
    y_sel = y[order]
    tp = int(y_sel.sum())
    precision = tp / k
    recall = tp / max(int(y.sum()), 1)
    base_rate = float(y.mean())
    return {
        "capacity_frac": capacity_frac,
        "n_selected": k,
        "churn_rate_in_selected": precision,
        "recall": recall,
        "lift": precision / base_rate if base_rate > 0 else float("nan"),
        "base_rate": base_rate,
    }


def f1_optimal_capacity(y: np.ndarray, score: np.ndarray) -> dict:
    """F1 最优的切点，用来和「承接量倒推」做对照（通常会大很多）。"""
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)
    precision = tp / k
    recall = tp / max(int(y.sum()), 1)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1.0), 0.0)
    best = int(np.argmax(f1))
    return {
        "capacity_frac": (best + 1) / len(y),
        "f1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "n_selected": int(best + 1),
    }


def cost_model(n_users_total: int, capacity_frac: float, cost_per_touch: float = 0.35) -> dict:
    """触达成本换算：成本 = 触达人数 × 单次成本。"""
    touched = int(n_users_total * capacity_frac)
    return {
        "touched": touched,
        "cost": touched * cost_per_touch,
        "cost_per_touch": cost_per_touch,
    }
