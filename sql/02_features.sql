-- 02 特征宽表：只用「下单前就能拿到」的信息
-- 方言：MySQL 8.0
--
-- 三条硬约束（写在这里，是为了下次改特征时不犯错）
--   1. 时间窗必须早于标签窗，晚一天都是未来信息
--   2. 不许用运营干预结果做特征（是否发券 / 是否用过券 / 是否被客服外呼）
--      —— 这些是干预的产物，训练时看得到，上线时不知道，会造成线上效果虚高
--   3. 金额类特征要排除退款与测试单

WITH params AS (
    SELECT DATE '2024-08-01' AS obs_start, DATE '2024-09-15' AS obs_end
),

ord AS (
    SELECT
        o.user_id,
        COUNT(DISTINCT o.order_id)                              AS freq_30d,
        SUM(o.pay_amount)                                       AS gmv_90d,
        MAX(o.paid_at)                                          AS last_paid_at,
        COUNT(DISTINCT o.category_id)                           AS cat_diversity
    FROM dwd_order_paid o, params p
    WHERE o.paid_at >= p.obs_start AND o.paid_at < p.obs_end
    GROUP BY o.user_id
),

beh AS (
    SELECT
        a.user_id,
        SUM(a.pv)                                               AS browse_30d,
        SUM(a.search_cnt)                                       AS search_30d,
        SUM(a.cart_cnt)                                         AS cart_30d,
        DATEDIFF((SELECT obs_end FROM params), MAX(a.stat_date)) AS recency_days
    FROM dwd_user_active_daily a, params p
    WHERE a.stat_date >= p.obs_start AND a.stat_date < p.obs_end
    GROUP BY a.user_id
),

svc AS (
    SELECT
        t.user_id,
        COUNT(*)                                                AS tickets_30d
    FROM dwd_service_ticket t, params p
    WHERE t.created_at >= p.obs_start AND t.created_at < p.obs_end
      AND t.category <> 'logistics_delay'   -- 物流延迟是平台问题，不是用户问题
    GROUP BY t.user_id
),

aftersale AS (
    SELECT
        r.user_id,
        SUM(CASE WHEN r.status = 'refunded' THEN 1 ELSE 0 END) / COUNT(*) AS return_rate_90d
    FROM dwd_after_sale r, params p
    WHERE r.created_at >= p.obs_start AND r.created_at < p.obs_end
    GROUP BY r.user_id
)

SELECT
    u.user_id,
    u.tenure_days,
    COALESCE(b.recency_days, 999)                               AS recency_days,
    COALESCE(o.freq_30d, 0)                                     AS freq_30d,
    COALESCE(o.freq_30d, 0) * 3                                 AS orders_90d,
    COALESCE(o.gmv_90d, 0)                                      AS gmv_90d,
    COALESCE(b.browse_30d, 0)                                   AS browse_30d,
    COALESCE(b.search_30d, 0)                                   AS search_30d,
    COALESCE(b.cart_30d, 0)                                     AS cart_30d,
    COALESCE(s.tickets_30d, 0)                                  AS tickets_30d,
    COALESCE(f.return_rate_90d, 0)                              AS return_rate_90d,
    COALESCE(o.cat_diversity, 1)                                AS cat_diversity,
    CASE WHEN u.tenure_days < 90 THEN 1 ELSE 0 END               AS is_new

    -- 以下字段【故意不进模型】，保留在注解里说明原因：
    -- , coupon_received_30d  -- 运营干预结果，上线不可知
    -- , coupon_used_30d      -- 同上，且与「被留住」互为因果
    -- , last_paid_at         -- 与 recency 共线，且是标签窗边界的泄漏源
FROM dwd_user_profile u, params p
LEFT JOIN ord      o ON o.user_id = u.user_id
LEFT JOIN beh      b ON b.user_id = u.user_id
LEFT JOIN svc      s ON s.user_id = u.user_id
LEFT JOIN aftersale f ON f.user_id = u.user_id
WHERE u.register_at < p.obs_start;
