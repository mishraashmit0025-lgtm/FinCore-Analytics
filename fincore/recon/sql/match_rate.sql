-- Share of bank lines cleared by each rule, plus what is still open.
WITH bank_lines AS (SELECT COUNT(*) AS n FROM bank),
cleared AS (
    SELECT m.rule, COUNT(DISTINCT l.record_id) AS bank_lines
    FROM matches m JOIN match_lines l ON l.match_id = m.match_id
    WHERE l.side = 'bank'
    GROUP BY m.rule
)
SELECT rule,
       bank_lines,
       ROUND(100.0 * bank_lines / (SELECT n FROM bank_lines), 2) AS pct_of_bank
FROM cleared
UNION ALL
SELECT 'open',
       (SELECT COUNT(*) FROM breaks WHERE side = 'bank' AND break_type = 'missing_ledger'),
       ROUND(100.0 * (SELECT COUNT(*) FROM breaks WHERE side = 'bank' AND break_type = 'missing_ledger')
             / (SELECT n FROM bank_lines), 2)
ORDER BY bank_lines DESC;
