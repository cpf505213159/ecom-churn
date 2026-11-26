"""同构数据生成器：按公司宽表的字段结构造一份可复现数据。

为什么需要它
------------
流失模型的真实宽表属于公司数据，不能外带。但**字段结构、标签口径、特征构造、
阈值倒推、成本核算**这些是分析师自己的资产，可以带。为了让这些资产可被验证，
这里按同样的结构造一份数据，并且——关键——**知道每个人的真实流失概率**。

有了真值，就能回答「AUC 0.78 到底是模型本事还是数据太easy」，
也能验证「按承接量倒推阈值」比「按 F1 最优点」到底差多少。

数据生成过程（DGP）
------------------
1. 每个用户有一个不可观测的流失倾向因子 f ~ N(0,1)
2. 流失概率 p = sigmoid(a + b·f)，据此抽标签
3. 观察到的特征是 f 的噪声代理：recency、频次、浏览、工单、退货……
4. **券**是运营干预的结果：它同时与 f 相关（运营爱给活跃用户发券），
   又和「后续是否下单」直接相关——这就是数据泄漏的来源，模型里必须砍掉。

运行：python -m src.gen
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20240915
N_USERS = 120_000

# 倾向到流失概率的映射：调 b 决定模型的天花板（AUC）
INTERCEPT = -2.05
SLOPE = 1.45


def build(n_users: int = N_USERS, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    # 1. 不可观测的倾向因子
    f = rng.normal(0, 1, n_users)

    # 2. 注册时长：新客判断依赖它，所以要在标签之前算出来
    tenure_days = np.clip(rng.gamma(3.0, 90, n_users), 1, 2000).round(0)

    # 3. 券：先有券，才有「被留住」这件事——这决定了泄漏的方向
    coupon_prob = 1 / (1 + np.exp(-(0.55 + 0.85 * f)))
    coupon_received_30d = (rng.random(n_users) < coupon_prob).astype(int)
    coupon_used_30d = ((coupon_received_30d == 1) & (rng.random(n_users) < 0.55)).astype(int)

    # 3. 真值标签：新客更容易流失（真实电商里这是最稳定的一条规律），
    #    券也确实留住了一部分人（干预效应），所以「用过券」与流失负相关
    is_new_flag = (tenure_days < 90).astype(float)
    logit = INTERCEPT + SLOPE * f + 0.95 * is_new_flag
    p_true = 1 / (1 + np.exp(-logit))
    p_obs = np.where(coupon_used_30d == 1, p_true * 0.30, p_true)
    churn = rng.random(n_users) < p_obs

    # 4. 特征：f 的噪声代理（噪声越大，AUC 越接近 0.5）
    recency_days = np.clip(30 - 7 * f + rng.normal(0, 4.5, n_users), 0, 180).round(0)
    freq_30d = np.clip(np.round(5 + 2.6 * f + rng.normal(0, 1.8, n_users)), 0, None)
    orders_90d = np.clip(np.round(freq_30d * 2.4 + rng.normal(0, 2, n_users)), 0, None)
    gmv_90d = np.round(np.exp(5.4 + 0.45 * f + rng.normal(0, 0.45, n_users)), 2)
    browse_30d = np.clip(np.round(45 + 22 * f + rng.normal(0, 16, n_users)), 0, None)
    search_30d = np.clip(np.round(10 + 5 * f + rng.normal(0, 5, n_users)), 0, None)
    cart_30d = np.clip(np.round(3 + 1.6 * f + rng.normal(0, 1.6, n_users)), 0, None)
    # 工单多的人更容易走：与 f 负相关
    tickets_30d = rng.poisson(np.exp(0.15 - 0.35 * f)).clip(0, 6)
    return_rate_90d = np.clip(rng.beta(2, 12, n_users) + 0.05 * (-f), 0, 1).round(3)
    cat_diversity = np.clip(np.round(2 + 1.2 * f + rng.normal(0, 1.2, n_users)), 1, None)

    df = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "tenure_days": tenure_days.astype(int),
        "recency_days": recency_days.astype(int),
        "freq_30d": freq_30d.astype(int),
        "orders_90d": orders_90d.astype(int),
        "gmv_90d": gmv_90d,
        "browse_30d": browse_30d.astype(int),
        "search_30d": search_30d.astype(int),
        "cart_30d": cart_30d.astype(int),
        "tickets_30d": tickets_30d.astype(int),
        "return_rate_90d": return_rate_90d,
        "cat_diversity": cat_diversity.astype(int),
        "coupon_received_30d": coupon_received_30d,
        "coupon_used_30d": coupon_used_30d,
        "is_new": (tenure_days < 90).astype(int),
        "churn": churn.astype(int),
    })

    truth = {
        "seed": SEED,
        "n_users": n_users,
        "churn_rate": float(churn.mean()),
        "coupon_retention_factor": 0.30,
        "intercept": INTERCEPT,
        "slope": SLOPE,
        "note": "churn 由不可观测因子 f 决定，特征是 f 的噪声代理；coupon_* 为泄漏特征",
    }
    return df, truth


def save(df: pd.DataFrame, truth: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "user_snapshot.csv", index=False, encoding="utf-8")
    (out / "ground_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    df, truth = build()
    save(df, truth, root / "outputs")
    print(f"生成 {len(df):,} 个用户｜流失率 {truth['churn_rate']:.1%}｜字段 {df.shape[1]} 个")
