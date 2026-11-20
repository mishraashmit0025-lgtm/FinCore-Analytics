SELECT counterparty,
       COUNT(*)              AS items,
       SUM(ABS(amount_paise)) AS open_value,
       MAX(age_days)         AS oldest_days
FROM breaks
WHERE counterparty <> ''
GROUP BY counterparty
ORDER BY open_value DESC
LIMIT 10;
