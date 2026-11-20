-- Absolute value of open breaks in paise, bucketed by age.
SELECT break_type,
       COUNT(*)                                                          AS items,
       SUM(CASE WHEN age_days <= 7 THEN ABS(amount_paise) ELSE 0 END)                   AS d0_7,
       SUM(CASE WHEN age_days BETWEEN 8 AND 30 THEN ABS(amount_paise) ELSE 0 END)       AS d8_30,
       SUM(CASE WHEN age_days BETWEEN 31 AND 90 THEN ABS(amount_paise) ELSE 0 END)      AS d31_90,
       SUM(CASE WHEN age_days > 90 THEN ABS(amount_paise) ELSE 0 END)                   AS d90_plus,
       SUM(ABS(amount_paise))                                            AS total
FROM breaks
GROUP BY break_type
ORDER BY total DESC;
