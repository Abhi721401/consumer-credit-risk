-- ============================================================================
-- analytical_queries.sql
-- Analytical queries for the Consumer Credit Risk database (SQLite).
-- Tables:
--   customers         -- full cleaned + feature-engineered population (30,000 rows)
--   scored_customers  -- held-out test set with model PD + risk band (7,500 rows)
-- ============================================================================

-- 1. Portfolio size and overall default rate
SELECT
    COUNT(*)                                   AS total_customers,
    SUM(default_next_month)                    AS total_defaulters,
    ROUND(100.0 * AVG(default_next_month), 2)  AS default_rate_pct
FROM customers;

-- 2. Average credit limit, overall and by default status
SELECT
    default_next_month,
    COUNT(*)               AS n_customers,
    ROUND(AVG(LIMIT_BAL), 0) AS avg_credit_limit,
    ROUND(AVG(UTIL_AVG), 3)  AS avg_utilization
FROM customers
GROUP BY default_next_month;

-- 3. Default rate and average predicted PD by risk segment (scored/test population)
SELECT
    risk_band,
    COUNT(*)                              AS n_customers,
    ROUND(AVG(predicted_pd), 4)           AS avg_predicted_pd,
    ROUND(100.0 * AVG(actual_default), 2) AS actual_default_rate_pct,
    ROUND(SUM(LIMIT_BAL), 0)              AS total_exposure
FROM scored_customers
GROUP BY risk_band
ORDER BY avg_predicted_pd;

-- 4. Delinquency analysis: default rate by number of delinquent months (of last 6)
SELECT
    N_MONTHS_DELINQUENT,
    COUNT(*)                                  AS n_customers,
    ROUND(100.0 * AVG(default_next_month), 2) AS default_rate_pct
FROM customers
GROUP BY N_MONTHS_DELINQUENT
ORDER BY N_MONTHS_DELINQUENT;

-- 5. High-risk customer identification (test population, Very High Risk band)
SELECT
    ID,
    LIMIT_BAL,
    predicted_pd,
    risk_band,
    N_MONTHS_DELINQUENT,
    UTIL_LATEST,
    actual_default
FROM scored_customers
WHERE risk_band = 'Very High Risk'
ORDER BY predicted_pd DESC
LIMIT 25;

-- 6. Portfolio summary by education level
SELECT
    EDUCATION,
    COUNT(*)                                  AS n_customers,
    ROUND(100.0 * AVG(default_next_month), 2) AS default_rate_pct,
    ROUND(AVG(LIMIT_BAL), 0)                  AS avg_credit_limit
FROM customers
GROUP BY EDUCATION
ORDER BY EDUCATION;

-- 7. Default rate by age band (bucketed in SQL via CASE)
SELECT
    CASE
        WHEN AGE < 25 THEN '<25'
        WHEN AGE < 35 THEN '25-34'
        WHEN AGE < 45 THEN '35-44'
        WHEN AGE < 55 THEN '45-54'
        ELSE '55+'
    END AS age_band,
    COUNT(*)                                  AS n_customers,
    ROUND(100.0 * AVG(default_next_month), 2) AS default_rate_pct
FROM customers
GROUP BY age_band
ORDER BY age_band;

-- 8. Utilization-stress population and their outcomes
SELECT
    UTILIZATION_STRESS_FLAG,
    COUNT(*)                                  AS n_customers,
    ROUND(100.0 * AVG(default_next_month), 2) AS default_rate_pct
FROM customers
GROUP BY UTILIZATION_STRESS_FLAG;

-- 9. Portfolio exposure concentration by risk band (share of total exposure)
SELECT
    risk_band,
    SUM(LIMIT_BAL)                                                    AS segment_exposure,
    ROUND(100.0 * SUM(LIMIT_BAL) / (SELECT SUM(LIMIT_BAL) FROM scored_customers), 2) AS pct_of_total_exposure
FROM scored_customers
GROUP BY risk_band
ORDER BY segment_exposure DESC;

-- 10. Model false negatives among high predicted-PD customers (governance/monitoring check)
SELECT
    COUNT(*) AS n_high_pd_customers,
    SUM(CASE WHEN actual_default = 0 THEN 1 ELSE 0 END) AS false_positives_in_band,
    SUM(CASE WHEN actual_default = 1 THEN 1 ELSE 0 END) AS true_positives_in_band
FROM scored_customers
WHERE risk_band IN ('High Risk', 'Very High Risk');
