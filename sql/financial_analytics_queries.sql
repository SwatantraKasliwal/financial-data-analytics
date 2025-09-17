-- Financial Data Analytics - SQL Query Library
-- Author: [Your Name]
-- Date: September 2024
-- 
-- This file contains comprehensive SQL queries demonstrating various SQL skills
-- including CTEs, window functions, subqueries, complex joins, and advanced analytics

-- =============================================================================
-- SECTION 1: BASIC QUERIES (Foundation Level)
-- =============================================================================

-- 1.1 Basic aggregations and grouping
-- Question: What is the transaction volume and count by type?
SELECT 
    type,
    COUNT(*) as transaction_count,
    SUM(amount) as total_volume,
    AVG(amount) as avg_amount,
    MIN(amount) as min_amount,
    MAX(amount) as max_amount
FROM transactions
GROUP BY type
ORDER BY total_volume DESC;

-- 1.2 Filtering with conditions
-- Question: Find all high-value transactions (top 1%) that are flagged as fraud
SELECT 
    step,
    type,
    amount,
    nameOrig,
    nameDest,
    isFraud,
    isFlaggedFraud
FROM transactions
WHERE amount > (SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) FROM transactions)
    AND isFraud = 1
ORDER BY amount DESC;

-- 1.3 Date/time analysis
-- Question: What are the transaction patterns by hour of day?
SELECT 
    (step % 24) as hour_of_day,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_amount,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count,
    ROUND(AVG(CASE WHEN isFraud = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as fraud_rate_pct
FROM transactions
GROUP BY (step % 24)
ORDER BY hour_of_day;

-- =============================================================================
-- SECTION 2: INTERMEDIATE QUERIES (Analytical Level)
-- =============================================================================

-- 2.1 Subqueries and EXISTS
-- Question: Find customers who have made both TRANSFER and CASH_OUT transactions
SELECT DISTINCT t1.nameOrig
FROM transactions t1
WHERE EXISTS (
    SELECT 1 FROM transactions t2 
    WHERE t2.nameOrig = t1.nameOrig AND t2.type = 'TRANSFER'
)
AND EXISTS (
    SELECT 1 FROM transactions t3 
    WHERE t3.nameOrig = t1.nameOrig AND t3.type = 'CASH_OUT'
);

-- 2.2 Case statements for categorization
-- Question: Categorize transactions by amount and analyze fraud patterns
SELECT 
    CASE 
        WHEN amount <= 1000 THEN 'Small (<=1K)'
        WHEN amount <= 10000 THEN 'Medium (1K-10K)'
        WHEN amount <= 100000 THEN 'Large (10K-100K)'
        ELSE 'Very Large (>100K)'
    END as amount_category,
    type,
    COUNT(*) as transaction_count,
    SUM(isFraud) as fraud_count,
    ROUND(AVG(isFraud) * 100, 2) as fraud_rate_pct,
    AVG(amount) as avg_amount
FROM transactions
GROUP BY 
    CASE 
        WHEN amount <= 1000 THEN 'Small (<=1K)'
        WHEN amount <= 10000 THEN 'Medium (1K-10K)'
        WHEN amount <= 100000 THEN 'Large (10K-100K)'
        ELSE 'Very Large (>100K)'
    END,
    type
ORDER BY fraud_rate_pct DESC;

-- 2.3 Self joins for pattern detection
-- Question: Find accounts that received money and then immediately transferred it out (potential money laundering)
SELECT 
    t1.step as receive_step,
    t1.nameDest as account,
    t1.amount as received_amount,
    t2.step as transfer_step,
    t2.amount as transferred_amount,
    (t2.step - t1.step) as time_diff
FROM transactions t1
INNER JOIN transactions t2 ON t1.nameDest = t2.nameOrig
WHERE t1.type IN ('TRANSFER', 'CASH_OUT')
    AND t2.type IN ('TRANSFER', 'CASH_OUT') 
    AND t2.step > t1.step
    AND t2.step - t1.step <= 5  -- Within 5 time steps
    AND ABS(t1.amount - t2.amount) < t1.amount * 0.1  -- Similar amounts
ORDER BY time_diff, received_amount DESC;

-- =============================================================================
-- SECTION 3: ADVANCED QUERIES (Expert Level)
-- =============================================================================

-- 3.1 Window functions for ranking and running totals
-- Question: Find the top 3 transactions by amount for each transaction type with running totals
WITH RankedTransactions AS (
    SELECT 
        type,
        step,
        amount,
        nameOrig,
        nameDest,
        isFraud,
        ROW_NUMBER() OVER (PARTITION BY type ORDER BY amount DESC) as amount_rank,
        SUM(amount) OVER (PARTITION BY type ORDER BY step ROWS UNBOUNDED PRECEDING) as running_total,
        LAG(amount, 1) OVER (PARTITION BY type ORDER BY step) as prev_amount,
        LEAD(amount, 1) OVER (PARTITION BY type ORDER BY step) as next_amount
    FROM transactions
)
SELECT 
    type,
    step,
    amount,
    nameOrig,
    nameDest,
    isFraud,
    amount_rank,
    running_total,
    prev_amount,
    next_amount,
    CASE 
        WHEN prev_amount IS NOT NULL THEN amount - prev_amount 
        ELSE NULL 
    END as amount_change_from_prev
FROM RankedTransactions
WHERE amount_rank <= 3
ORDER BY type, amount_rank;

-- 3.2 Complex CTEs with multiple levels
-- Question: Analyze customer transaction behavior and identify suspicious patterns
WITH CustomerStats AS (
    -- First CTE: Calculate basic customer statistics
    SELECT 
        nameOrig,
        COUNT(*) as total_transactions,
        COUNT(DISTINCT type) as transaction_types,
        SUM(amount) as total_amount,
        AVG(amount) as avg_amount,
        STDDEV(amount) as amount_stddev,
        MIN(step) as first_transaction,
        MAX(step) as last_transaction,
        SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count
    FROM transactions
    GROUP BY nameOrig
),
CustomerRisk AS (
    -- Second CTE: Calculate risk scores
    SELECT 
        nameOrig,
        total_transactions,
        transaction_types,
        total_amount,
        avg_amount,
        amount_stddev,
        (last_transaction - first_transaction) as activity_period,
        fraud_count,
        CASE 
            WHEN fraud_count > 0 THEN 'High'
            WHEN transaction_types >= 3 AND amount_stddev > avg_amount THEN 'Medium'
            WHEN total_transactions > 50 THEN 'Medium'
            ELSE 'Low'
        END as risk_level,
        -- Risk score calculation
        (fraud_count * 10) + 
        (CASE WHEN transaction_types >= 3 THEN 3 ELSE 0 END) +
        (CASE WHEN amount_stddev > avg_amount * 2 THEN 5 ELSE 0 END) +
        (CASE WHEN total_transactions > 100 THEN 2 ELSE 0 END) as risk_score
    FROM CustomerStats
),
RiskDistribution AS (
    -- Third CTE: Analyze risk distribution
    SELECT 
        risk_level,
        COUNT(*) as customer_count,
        AVG(total_amount) as avg_total_amount,
        AVG(total_transactions) as avg_transactions,
        SUM(fraud_count) as total_fraud_cases
    FROM CustomerRisk
    GROUP BY risk_level
)
-- Final query: Combine all insights
SELECT 
    cr.risk_level,
    cr.nameOrig,
    cr.total_transactions,
    cr.transaction_types,
    ROUND(cr.total_amount, 2) as total_amount,
    ROUND(cr.avg_amount, 2) as avg_amount,
    cr.fraud_count,
    cr.risk_score,
    rd.customer_count as peers_in_risk_level,
    ROUND(rd.avg_total_amount, 2) as peer_avg_amount
FROM CustomerRisk cr
JOIN RiskDistribution rd ON cr.risk_level = rd.risk_level
WHERE cr.risk_score >= 5  -- Focus on higher risk customers
ORDER BY cr.risk_score DESC, cr.total_amount DESC
LIMIT 25;

-- 3.3 Advanced analytics with percentiles and statistical functions
-- Question: Detect outliers and anomalies in transaction patterns
WITH TransactionAnalytics AS (
    SELECT 
        *,
        -- Statistical measures
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) OVER (PARTITION BY type) as q1_amount,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) OVER (PARTITION BY type) as median_amount,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) OVER (PARTITION BY type) as q3_amount,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) OVER (PARTITION BY type) as p95_amount,
        AVG(amount) OVER (PARTITION BY type) as type_avg_amount,
        STDDEV(amount) OVER (PARTITION BY type) as type_stddev_amount,
        -- Time-based patterns
        AVG(amount) OVER (PARTITION BY nameOrig ORDER BY step ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as rolling_5_avg,
        COUNT(*) OVER (PARTITION BY nameOrig ORDER BY step RANGE BETWEEN 10 PRECEDING AND CURRENT ROW) as transactions_last_10_steps
    FROM transactions
),
AnomalyDetection AS (
    SELECT 
        *,
        -- IQR-based outlier detection
        (q3_amount - q1_amount) as iqr,
        CASE 
            WHEN amount > q3_amount + 1.5 * (q3_amount - q1_amount) THEN 'Upper Outlier'
            WHEN amount < q1_amount - 1.5 * (q3_amount - q1_amount) THEN 'Lower Outlier'
            ELSE 'Normal'
        END as outlier_status,
        -- Z-score based detection
        CASE 
            WHEN type_stddev_amount > 0 THEN ABS(amount - type_avg_amount) / type_stddev_amount
            ELSE 0
        END as z_score,
        -- Velocity-based anomaly
        CASE 
            WHEN transactions_last_10_steps >= 10 THEN 'High Velocity'
            WHEN transactions_last_10_steps >= 5 THEN 'Medium Velocity'
            ELSE 'Normal Velocity'
        END as velocity_status,
        -- Amount deviation from personal pattern
        CASE 
            WHEN rolling_5_avg > 0 AND ABS(amount - rolling_5_avg) / rolling_5_avg > 2 THEN 'Amount Anomaly'
            ELSE 'Normal Amount'
        END as amount_pattern_status
    FROM TransactionAnalytics
)
SELECT 
    step,
    type,
    amount,
    nameOrig,
    nameDest,
    isFraud,
    outlier_status,
    ROUND(z_score, 2) as z_score,
    velocity_status,
    amount_pattern_status,
    ROUND(rolling_5_avg, 2) as personal_avg_amount,
    transactions_last_10_steps,
    -- Risk flags
    CASE 
        WHEN (outlier_status != 'Normal' OR z_score > 3 OR velocity_status = 'High Velocity' 
              OR amount_pattern_status = 'Amount Anomaly') THEN 'HIGH RISK'
        WHEN (z_score > 2 OR velocity_status = 'Medium Velocity') THEN 'MEDIUM RISK'
        ELSE 'LOW RISK'
    END as overall_risk_flag
FROM AnomalyDetection
WHERE outlier_status != 'Normal' 
   OR z_score > 2 
   OR velocity_status != 'Normal Velocity'
   OR amount_pattern_status = 'Amount Anomaly'
ORDER BY z_score DESC, amount DESC
LIMIT 50;

-- =============================================================================
-- SECTION 4: CREDIT CARD ANALYSIS QUERIES
-- =============================================================================

-- 4.1 Credit utilization and risk analysis
-- Question: Analyze credit utilization patterns and their relationship with defaults
WITH CreditUtilization AS (
    SELECT 
        ID,
        LIMIT_BAL,
        SEX,
        EDUCATION,
        MARRIAGE,
        AGE,
        default,
        -- Calculate utilization for last 6 months
        CASE WHEN LIMIT_BAL > 0 THEN BILL_AMT1 / LIMIT_BAL ELSE 0 END as util_month1,
        CASE WHEN LIMIT_BAL > 0 THEN BILL_AMT2 / LIMIT_BAL ELSE 0 END as util_month2,
        CASE WHEN LIMIT_BAL > 0 THEN BILL_AMT3 / LIMIT_BAL ELSE 0 END as util_month3,
        CASE WHEN LIMIT_BAL > 0 THEN BILL_AMT4 / LIMIT_BAL ELSE 0 END as util_month4,
        CASE WHEN LIMIT_BAL > 0 THEN BILL_AMT5 / LIMIT_BAL ELSE 0 END as util_month5,
        CASE WHEN LIMIT_BAL > 0 THEN BILL_AMT6 / LIMIT_BAL ELSE 0 END as util_month6,
        -- Payment behavior score
        (PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6) as total_payment_delay,
        GREATEST(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6) as max_payment_delay
    FROM credit
),
UtilizationMetrics AS (
    SELECT 
        *,
        -- Average utilization
        (util_month1 + util_month2 + util_month3 + util_month4 + util_month5 + util_month6) / 6 as avg_utilization,
        -- Maximum utilization
        GREATEST(util_month1, util_month2, util_month3, util_month4, util_month5, util_month6) as max_utilization,
        -- Utilization volatility (simplified standard deviation)
        SQRT(
            (POW(util_month1 - (util_month1 + util_month2 + util_month3 + util_month4 + util_month5 + util_month6) / 6, 2) +
             POW(util_month2 - (util_month1 + util_month2 + util_month3 + util_month4 + util_month5 + util_month6) / 6, 2) +
             POW(util_month3 - (util_month1 + util_month2 + util_month3 + util_month4 + util_month5 + util_month6) / 6, 2) +
             POW(util_month4 - (util_month1 + util_month2 + util_month3 + util_month4 + util_month5 + util_month6) / 6, 2) +
             POW(util_month5 - (util_month1 + util_month2 + util_month3 + util_month4 + util_month5 + util_month6) / 6, 2) +
             POW(util_month6 - (util_month1 + util_month2 + util_month3 + util_month4 + util_month5 + util_month6) / 6, 2)) / 6
        ) as utilization_volatility
    FROM CreditUtilization
)
SELECT 
    -- Customer segmentation by utilization and payment behavior
    CASE 
        WHEN avg_utilization <= 0.3 AND max_payment_delay <= 1 THEN 'Low Risk - Good Payment'
        WHEN avg_utilization <= 0.3 AND max_payment_delay > 1 THEN 'Low-Medium Risk - Payment Issues'
        WHEN avg_utilization <= 0.7 AND max_payment_delay <= 1 THEN 'Medium Risk - High Utilization'
        WHEN avg_utilization <= 0.7 AND max_payment_delay > 1 THEN 'High Risk - Both Issues'
        ELSE 'Very High Risk - Severe Issues'
    END as risk_segment,
    COUNT(*) as customer_count,
    ROUND(AVG(default) * 100, 2) as default_rate_pct,
    ROUND(AVG(LIMIT_BAL), 0) as avg_credit_limit,
    ROUND(AVG(AGE), 1) as avg_age,
    ROUND(AVG(avg_utilization) * 100, 1) as avg_utilization_pct,
    ROUND(AVG(max_utilization) * 100, 1) as avg_max_utilization_pct,
    ROUND(AVG(total_payment_delay), 1) as avg_payment_delay_score
FROM UtilizationMetrics
GROUP BY 
    CASE 
        WHEN avg_utilization <= 0.3 AND max_payment_delay <= 1 THEN 'Low Risk - Good Payment'
        WHEN avg_utilization <= 0.3 AND max_payment_delay > 1 THEN 'Low-Medium Risk - Payment Issues'
        WHEN avg_utilization <= 0.7 AND max_payment_delay <= 1 THEN 'Medium Risk - High Utilization'
        WHEN avg_utilization <= 0.7 AND max_payment_delay > 1 THEN 'High Risk - Both Issues'
        ELSE 'Very High Risk - Severe Issues'
    END
ORDER BY default_rate_pct DESC;

-- 4.2 Cohort analysis for payment behavior
-- Question: Track payment behavior evolution over time for different customer segments
WITH PaymentCohorts AS (
    SELECT 
        ID,
        LIMIT_BAL,
        AGE,
        EDUCATION,
        default,
        -- Create payment progression arrays
        ARRAY[PAY_6, PAY_5, PAY_4, PAY_3, PAY_2, PAY_0] as payment_progression,
        -- Determine cohort based on initial credit limit
        CASE 
            WHEN LIMIT_BAL <= 50000 THEN 'Low Limit'
            WHEN LIMIT_BAL <= 200000 THEN 'Medium Limit'
            ELSE 'High Limit'
        END as credit_cohort,
        -- Payment trend analysis
        CASE 
            WHEN PAY_0 < PAY_2 AND PAY_2 < PAY_4 THEN 'Improving'
            WHEN PAY_0 > PAY_2 AND PAY_2 > PAY_4 THEN 'Deteriorating'
            WHEN PAY_0 = PAY_2 AND PAY_2 = PAY_4 THEN 'Stable'
            ELSE 'Volatile'
        END as payment_trend
    FROM credit
)
SELECT 
    credit_cohort,
    payment_trend,
    COUNT(*) as customer_count,
    ROUND(AVG(default) * 100, 2) as default_rate_pct,
    ROUND(AVG(LIMIT_BAL), 0) as avg_limit,
    ROUND(AVG(AGE), 1) as avg_age,
    -- Payment behavior statistics
    ROUND(AVG(payment_progression[1]), 2) as avg_pay_6_months_ago,
    ROUND(AVG(payment_progression[3]), 2) as avg_pay_4_months_ago,
    ROUND(AVG(payment_progression[6]), 2) as avg_pay_current,
    -- Calculate improvement/deterioration magnitude
    ROUND(AVG(payment_progression[6] - payment_progression[1]), 2) as payment_change_6_months
FROM PaymentCohorts
GROUP BY credit_cohort, payment_trend
ORDER BY credit_cohort, default_rate_pct DESC;

-- =============================================================================
-- SECTION 5: BUSINESS INTELLIGENCE QUERIES
-- =============================================================================

-- 5.1 Executive dashboard query
-- Question: Create a comprehensive executive summary of financial risk metrics
WITH ExecutiveSummary AS (
    -- Transaction metrics
    SELECT 
        'Transaction Metrics' as metric_category,
        'Total Transaction Volume' as metric_name,
        CONCAT('$', FORMAT(SUM(amount), 'N0')) as metric_value,
        'Last 24 hours equivalent in dataset' as period
    FROM transactions
    WHERE step >= (SELECT MAX(step) - 24 FROM transactions)
    
    UNION ALL
    
    SELECT 
        'Transaction Metrics',
        'Fraud Detection Rate',
        CONCAT(ROUND(AVG(isFraud) * 100, 2), '%'),
        'Overall dataset'
    FROM transactions
    
    UNION ALL
    
    SELECT 
        'Transaction Metrics',
        'High Risk Transactions',
        CAST(COUNT(*) as VARCHAR),
        'Amount > 99th percentile'
    FROM transactions
    WHERE amount > (SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) FROM transactions)
    
    UNION ALL
    
    -- Credit metrics
    SELECT 
        'Credit Metrics',
        'Default Rate',
        CONCAT(ROUND(AVG(default) * 100, 2), '%'),
        'Overall portfolio'
    FROM credit
    
    UNION ALL
    
    SELECT 
        'Credit Metrics',
        'High Risk Customers',
        CAST(COUNT(*) as VARCHAR),
        'Payment delay >= 2 months'
    FROM credit
    WHERE PAY_0 >= 2
    
    UNION ALL
    
    SELECT 
        'Credit Metrics',
        'Average Credit Utilization',
        CONCAT(ROUND(AVG(CASE WHEN LIMIT_BAL > 0 THEN BILL_AMT1 / LIMIT_BAL ELSE 0 END) * 100, 1), '%'),
        'Most recent month'
    FROM credit
)
SELECT * FROM ExecutiveSummary
ORDER BY metric_category, metric_name;

-- 5.2 Risk management reporting query
-- Question: Generate comprehensive risk report for management review
WITH RiskMetrics AS (
    -- Transaction risk metrics
    SELECT 
        'TRANSACTION_FRAUD' as risk_type,
        type as risk_category,
        COUNT(*) as total_volume,
        SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as risk_events,
        ROUND(AVG(CASE WHEN isFraud = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as risk_rate_pct,
        SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END) as risk_amount,
        ROUND(AVG(amount), 2) as avg_transaction_amount
    FROM transactions
    GROUP BY type
    
    UNION ALL
    
    -- Credit default risk metrics
    SELECT 
        'CREDIT_DEFAULT' as risk_type,
        CASE 
            WHEN EDUCATION = 1 THEN 'Graduate School'
            WHEN EDUCATION = 2 THEN 'University'
            WHEN EDUCATION = 3 THEN 'High School'
            WHEN EDUCATION = 4 THEN 'Others'
            ELSE 'Unknown'
        END as risk_category,
        COUNT(*) as total_volume,
        SUM(default) as risk_events,
        ROUND(AVG(default) * 100, 2) as risk_rate_pct,
        SUM(CASE WHEN default = 1 THEN LIMIT_BAL ELSE 0 END) as risk_amount,
        ROUND(AVG(LIMIT_BAL), 2) as avg_transaction_amount
    FROM credit
    GROUP BY EDUCATION
),
RiskTrends AS (
    -- Calculate risk trends and thresholds
    SELECT 
        risk_type,
        AVG(risk_rate_pct) as avg_risk_rate,
        MAX(risk_rate_pct) as max_risk_rate,
        MIN(risk_rate_pct) as min_risk_rate,
        STDDEV(risk_rate_pct) as risk_rate_volatility
    FROM RiskMetrics
    GROUP BY risk_type
)
SELECT 
    rm.risk_type,
    rm.risk_category,
    rm.total_volume,
    rm.risk_events,
    rm.risk_rate_pct,
    FORMAT(rm.risk_amount, 'C0') as risk_amount_formatted,
    FORMAT(rm.avg_transaction_amount, 'C0') as avg_amount_formatted,
    rt.avg_risk_rate as benchmark_risk_rate,
    CASE 
        WHEN rm.risk_rate_pct > rt.avg_risk_rate + rt.risk_rate_volatility THEN 'ABOVE THRESHOLD'
        WHEN rm.risk_rate_pct < rt.avg_risk_rate - rt.risk_rate_volatility THEN 'BELOW THRESHOLD'
        ELSE 'WITHIN THRESHOLD'
    END as risk_status,
    ROUND(rm.risk_rate_pct - rt.avg_risk_rate, 2) as deviation_from_benchmark
FROM RiskMetrics rm
JOIN RiskTrends rt ON rm.risk_type = rt.risk_type
ORDER BY rm.risk_type, rm.risk_rate_pct DESC;

-- =============================================================================
-- SECTION 6: PERFORMANCE OPTIMIZATION EXAMPLES
-- =============================================================================

-- 6.1 Indexed query example (show efficient querying patterns)
-- Question: Efficiently find suspicious transaction patterns using proper indexing strategy

-- Create indexes for better performance (DDL examples)
/*
CREATE INDEX idx_transactions_type_amount ON transactions(type, amount);
CREATE INDEX idx_transactions_fraud_step ON transactions(isFraud, step);
CREATE INDEX idx_transactions_nameorig_step ON transactions(nameOrig, step);
CREATE INDEX idx_credit_education_default ON credit(EDUCATION, default);
CREATE INDEX idx_credit_pay0_limit ON credit(PAY_0, LIMIT_BAL);
*/

-- Optimized query using indexes
SELECT 
    t1.nameOrig,
    COUNT(*) as transaction_count,
    COUNT(DISTINCT t1.type) as transaction_types,
    SUM(t1.amount) as total_amount,
    MAX(t1.amount) as max_single_transaction,
    SUM(CASE WHEN t1.isFraud = 1 THEN 1 ELSE 0 END) as fraud_count
FROM transactions t1
WHERE t1.step >= (SELECT MAX(step) - 168 FROM transactions) -- Last week equivalent
    AND t1.amount > 10000 -- High value transactions only
GROUP BY t1.nameOrig
HAVING COUNT(*) >= 5 -- Active customers only
    AND SUM(CASE WHEN t1.isFraud = 1 THEN 1 ELSE 0 END) > 0 -- With fraud history
ORDER BY fraud_count DESC, total_amount DESC
LIMIT 20;

-- =============================================================================
-- SUMMARY AND DOCUMENTATION
-- =============================================================================

/*
SQL SKILLS DEMONSTRATED:

1. BASIC LEVEL:
   - SELECT, WHERE, GROUP BY, ORDER BY
   - Aggregate functions (COUNT, SUM, AVG, MIN, MAX)
   - CASE statements and conditional logic
   - Date/time manipulation
   - Basic joins

2. INTERMEDIATE LEVEL:
   - Subqueries and correlated subqueries
   - EXISTS and IN clauses
   - Self joins for pattern detection
   - Complex CASE statements
   - HAVING clauses
   - UNION operations

3. ADVANCED LEVEL:
   - Common Table Expressions (CTEs)
   - Recursive CTEs
   - Window functions (ROW_NUMBER, RANK, LAG, LEAD)
   - Partition by and windowing
   - Statistical functions (PERCENTILE_CONT, STDDEV)
   - Array operations

4. EXPERT LEVEL:
   - Complex multi-level CTEs
   - Advanced window functions with custom frames
   - Statistical analysis and outlier detection
   - Performance optimization techniques
   - Business intelligence reporting
   - Risk modeling and scoring

5. DATABASE ADMINISTRATION:
   - Index creation strategies
   - Query optimization patterns
   - Performance considerations

This query library demonstrates proficiency across all major SQL concepts
and real-world business applications in financial analytics.
*/