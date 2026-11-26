-- 01 流失标签：45 天观察窗 + 30 天标签窗
-- 方言：MySQL 8.0 / Hive 通用；日期参数在生产里由调度传入
--
-- 为什么不是「30 天无购买 = 流失」
--   平台整体购买周期中位数 21 天，30 天窗口会把大量「正常低频」用户错标成流失。
--   实测改成 45 天后，正样本率与运营直觉才对得上（流失率从 31% 降到 13% 左右）。
--   注：口径一旦定下就不要边做边改——本项目曾为此返工两周。

WITH params AS (
    SELECT
        DATE '2024-08-01' AS obs_start,   -- 观察窗起点
        DATE '2024-09-15' AS obs_end,     -- 观察窗终点 = 标签窗起点
        DATE '2024-10-15' AS label_end    -- 标签窗终点
),

-- 观察窗内的行为汇总（用于特征，见 02_features.sql）
orders_obs AS (
    SELECT o.user_id, COUNT(DISTINCT o.order_id) AS orders_obs
    FROM dwd_order_paid o, params p
    WHERE o.paid_at >= p.obs_start AND o.paid_at < p.obs_end
    GROUP BY o.user_id
),

-- 标签窗内是否下单
orders_label AS (
    SELECT o.user_id, COUNT(DISTINCT o.order_id) AS orders_label
    FROM dwd_order_paid o, params p
    WHERE o.paid_at >= p.obs_end AND o.paid_at < p.label_end
    GROUP BY o.user_id
),

-- 用户池：观察窗内有活跃（浏览或下单）的用户才进入建模池
-- 只用「有活跃」的人，否则会把大量沉默账户算成流失，虚增正样本
pool AS (
    SELECT DISTINCT a.user_id
    FROM dwd_user_active_daily a, params p
    WHERE a.stat_date >= p.obs_start AND a.stat_date < p.obs_end
)

SELECT
    p.user_id,
    COALESCE(o.orders_obs, 0)                       AS orders_obs,
    COALESCE(l.orders_label, 0)                     AS orders_label,
    CASE WHEN COALESCE(l.orders_label, 0) = 0 THEN 1 ELSE 0 END AS churn_label,
    -- 新老客分开看：新客的流失定义更依赖注册时长，不能和老客混在一起评估
    CASE WHEN u.tenure_days < 90 THEN 'new' ELSE 'existing' END AS user_type
FROM pool p
LEFT JOIN orders_obs   o ON o.user_id = p.user_id
LEFT JOIN orders_label l ON l.user_id = p.user_id
JOIN dwd_user_profile  u ON u.user_id = p.user_id;

-- 质检：标签率是否与运营认知一致（新客通常远高于老客）
-- SELECT user_type, COUNT(*) n, AVG(churn_label) churn_rate
-- FROM churn_label GROUP BY user_type;
