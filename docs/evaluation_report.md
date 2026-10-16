# Synthetic Evaluation Report

## Reproducible configuration

This report contains computed synthetic results, not estimates or OLG performance claims.

- Development: 300 players, 6,000 events, seed 142.
- Held-out evaluation: 300 different synthetic players, 6,000 events, seed 1042.
- Review budget: top 50 accounts for every detector/domain comparison.
- Fixed score threshold: 60.
- Evaluation labels were applied only after scoring.

## Held-out fixed-budget results

| Detector | Domain | Precision@50 | Recall@50 | False positives | False negatives |
|---|---|---:|---:|---:|---:|
| Fixed Threshold Baseline | AML/fraud | 12.00% | 37.50% | 44 | 10 |
| Fixed Threshold Baseline | Safer gambling | 12.00% | 42.86% | 44 | 8 |
| Single-Channel Detection | AML/fraud | 12.00% | 37.50% | 44 | 10 |
| Single-Channel Detection | Safer gambling | 12.00% | 42.86% | 44 | 8 |
| Cross-Channel Detection | AML/fraud | 12.00% | 37.50% | 44 | 10 |
| Cross-Channel Detection | Safer gambling | 12.00% | 42.86% | 44 | 8 |
| Cross-Channel + Network Intelligence | AML/fraud | 18.00% | 56.25% | 41 | 7 |
| Cross-Channel + Network Intelligence | Safer gambling | 12.00% | 42.86% | 44 | 8 |

The fixed threshold of 60 generated 7 baseline AML alerts and 1 baseline safer-gambling alert. The other detector variants generated zero alerts at that threshold in this run. That is a configuration limitation, not evidence of safety; fixed-budget metrics provide the fair workload comparison.

## Scenario findings

At top 50, cross-channel plus network intelligence detected 3/3 coordinated-account cases and 4/4 unusual-geography cases in the AML synthetic labels. It detected 0/3 abrupt-change and 0/3 distributed-cross-channel cases in that AML view. For safer gambling it detected 3/4 increasing-loss/deposit cases, 3/4 spending-escalation cases, and 0/3 each for abrupt-change and distributed-cross-channel cases. These misses are material.

## Channel and visibility sensitivity

For cross-channel plus network AML, recall among positive players observed in each channel was 90.00% for land casino, 57.14% for online casino, and 60.00% for sportsbook. Safer-gambling channel recall was 66.67%, 46.15%, and 54.55%. Players can occur in multiple channel rows.

In the Red vs Blue visibility experiment, Blue Detector V2 measured Recall@50 of 54.17% with complete visibility, 58.33% with the sampled partial-visibility transformation, 45.83% with single-channel visibility, and 41.67% with uncertain linkage. The non-monotonic partial result is a measured ranking effect, not a claimed improvement from missing data.

## Behavioral adaptation arena

Held-out Recall@50 for Blue Detector V2 was 54.17% at Level 0, 50.00% at Level 1, 54.17% at Level 2, and 29.17% at Level 3. At Level 3 it missed 17 of 24 positive players. Simple baseline recall at the same levels was 50.00%, 41.67%, 45.83%, and 37.50%. No detector dominated every stress level.

## Sensitivity cohorts

- The held-out population contained no account under 90 days tenure, so cold-start performance is unavailable.
- Three benign high-spending controls existed; none entered the top-50 AML or safer-gambling queue.
- The broad non-injected geographic-change cohort contained 119 players; 36 AML top-50 alerts in that cohort were false positives.
- Uncertain identity linkage reduced cross-channel-plus-network AML Recall@50 from 56.25% to 43.75% in the Phase 11 sensitivity path.

## Fairness audit

Age band and region were used only after scoring. The maximum-to-minimum nonzero flag-rate ratio was 1.75 across age bands and 3.17 across regions for AML; for safer gambling it was 1.62 and 4.34. Every disparity table carried a small-sample warning. Ratios are descriptive and unstable. Equal flag rates would not necessarily mean fair outcomes, and these synthetic results support no fairness claim.

## Limitations

The evaluation reflects generator assumptions, small positive counts, one held-out seed, and simplified labels. It does not measure real harm, fraud, operational cost, calibration, causal fairness, or production drift. See [limitations](limitations.md).
