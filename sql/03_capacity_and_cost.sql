-- 03 阈值与成本：按运营承接量切名单，不按 F1 最优点
-- 方言：MySQL 8.0
--
-- 核心观点：阈值是业务参数，不是统计参数。
--   F1 最优会把名单放大到全量的 40%~50%，运营根本发不完；
--   正确做法是先拿到「本月最多能触达多少人」这个硬约束，再回头检查
--   这个切点的精确率是否可接受。
--
-- 步骤：给每个用户打分（score 由模型离线产出，写回 dwd_churn_score），
--       按分数倒序排名，取前 capacity 名。

WITH params AS (
    SELECT 800000 AS capacity        -- 运营给的硬约束：本月最多触达 80 万人
),

scored AS (
    SELECT
        s.user_id,
        s.score,
        l.churn_label,
        ROW_NUMBER() OVER (ORDER BY s.score DESC) AS rn,
        COUNT(*) OVER ()                          AS total_users
    FROM dwd_churn_score s
    JOIN churn_label l ON l.user_id = s.user_id
)

SELECT
    -- 1) 切点本身
    MAX(CASE WHEN rn = (SELECT capacity FROM params) THEN score END)          AS threshold_score,
    -- 2) 名单质量
    SUM(CASE WHEN rn <= (SELECT capacity FROM params) THEN 1 ELSE 0 END)      AS n_selected,
    AVG(CASE WHEN rn <= (SELECT capacity FROM params) THEN churn_label END)   AS churn_rate_in_list,
    -- 3) 召回：名单覆盖了多少真实流失用户
    SUM(CASE WHEN rn <= (SELECT capacity FROM params) THEN churn_label ELSE 0 END)
        / SUM(churn_label)                                                    AS recall,
    -- 4) 与全量触达的对比：成本按 0.35 元/人
    SUM(CASE WHEN rn <= (SELECT capacity FROM params) THEN 1 ELSE 0 END) * 0.35 AS cost_selected,
    SUM(churn_label) * 0 + MAX(total_users) * 0.35                             AS cost_full
FROM scored;

-- 分层的效果核对：名单里各分数段的实际流失率，用于上线前的直觉校验
-- 如果最高分段（top 5%）的流失率没有明显高于 base rate，说明特征或口径有问题
SELECT
    CASE
        WHEN rn <= total_users * 0.05 THEN 'top 5%'
        WHEN rn <= total_users * 0.15 THEN 'top 15%'
        WHEN rn <= total_users * 0.30 THEN 'top 30%'
        ELSE 'rest'
    END                              AS band,
    COUNT(*)                         AS users,
    AVG(churn_label)                 AS actual_churn_rate
FROM scored
GROUP BY band
ORDER BY MIN(rn);
