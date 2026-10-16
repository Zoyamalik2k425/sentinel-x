# Sentinel X: likely technical judge questions

These answers describe the current code, not an intended future version. Begin with the short answer; use the technical detail if a judge asks a follow-up.

## 1. Is this actually machine learning, or is it just rules?

**Short answer:** Both. The main dashboard uses a trained unsupervised Isolation Forest and trained KMeans peer model, plus a statistical change detector and transparent weighted rules for the final review indices.

**Technical detail:** `ml/anomaly_detection.py` fits scikit-learn `IsolationForest`; `ml/peer_analysis.py` fits KMeans and training-only robust peer distributions. `ml/behavioral_change.py` uses rolling median/MAD and CUSUM rather than fitting labels. `scoring.py` then combines normalized signals with configured weights. The simpler validation/Arena detectors in `signals.py` and `models.py` are heuristic benchmarks, not the full dashboard pipeline.

## 2. Are the AML and safer-gambling scores probabilities?

**Short answer:** No. They are 0-100 review indices.

**Technical detail:** Available signals are graded from raw strength and population percentile, then combined through normalized configured weights. The system does not perform probability calibration and explicitly says so in case narratives. Isolation Forest severity is also an empirical training-score percentile, not a fraud probability.

## 3. What are the exact final score weights?

**Short answer:** AML is 40% transaction anomaly, 30% funds movement, 20% network, 4% geography, 3% behavioral change, and 3% cross-channel inconsistency. Safer Gambling is 25% deposit escalation, 25% reliable net-loss increase, 15% session increase, 20% wagering-intensity change, and 15% historical-pattern change.

**Technical detail:** Missing signals are not assigned zero. Their weights are removed from the denominator, contributions are normalized over available evidence, and evidence-weight coverage is reported separately. Each contribution is `100 * graded signal * normalized available weight`.

## 4. Why renormalize weights when evidence is missing? Could that inflate a score?

**Short answer:** It avoids calling unknown evidence safe, but yes, a strong partial-evidence score can be high. That is why coverage is displayed separately and prominently.

**Technical detail:** A score answers “how concerning is the available evidence?” Coverage answers “how much configured evidence was available?” The worked example scores 98 for safer gambling with only 40% evidence-weight coverage. Production policy could add minimum-evidence gates, but the prototype intentionally does not hide missingness inside the score.

## 5. How do you prevent synthetic labels from leaking into the model?

**Short answer:** Labels live in a separate DataFrame and operational modules reject label columns.

**Technical detail:** Profiles and channel events do not contain scenario columns. Feature engineering, model fitting, scoring, network analysis, and copilot boundaries check forbidden names and raise errors. `scenario_labels` enters only threshold/evaluation or stress-targeting functions after label-free scores exist. Tests deliberately pass leaked columns and expect rejection.

## 6. How do you prevent future information leakage?

**Short answer:** Features for time T use records at or before T, and training/validation/test are split chronologically.

**Technical detail:** Feature masks use `timestamp <= current timestamp`; same-time behavioral-change rows share the same prior state. Training uses the earliest 60% of distinct times, validation the next 20%, and test the final 20%. Tests add future transactions and verify earlier feature values do not change.

## 7. How are the three channels joined if their data quality differs?

**Short answer:** They are normalized into a common event schema, while channel, source IDs, observation flags, and identity confidence are preserved.

**Technical detail:** Buy-ins map to deposits, cash-outs to withdrawals, settlements to payouts, and observed wagering to wagers. Unobserved land wager/payout amounts become missing, not zero. Channel-specific and unified summaries are both retained.

## 8. What happens when identity matching is uncertain?

**Short answer:** The records remain separate and are shown as probable, unlinked, or conflicting source accounts.

**Technical detail:** Only a unique asserted identity with confidence at least 0.95 receives a canonical `PLAYER:` entity. A probable match retains a proposed ID for review but gets its own `SOURCE:` entity. Conflicts are quarantined. Identity confidence is displayed but does not multiply risk.

## 9. How do you avoid double counting the same transaction?

**Short answer:** The integration layer deduplicates exact source records and duplicate representations while retaining distinct financial legs.

**Technical detail:** It first keys on channel and source-event ID, then on transaction reference, direction, and rounded amount. The best-observed, highest-confidence, most-complete representation is kept. A deposit and withdrawal with the same reference remain distinct because direction differs.

## 10. What makes a network relationship suspicious?

**Short answer:** No single shared identifier establishes fraud. The graph treats relationships as investigation leads and flags clusters only when multiple kinds or repeated patterns of evidence combine.

**Technical detail:** Edges can include shared devices, payment tokens, at least three overlapping hourly activity windows, and confirmed ownership. Ownership adds relationship strength but zero investigation strength. A ring flag requires at least three accounts, multiple identifier edges, and repeated correlated activity or strong multi-identifier structure.

## 11. Could a family sharing one device be incorrectly flagged?

**Short answer:** It can create an explainable edge, but one shared device alone does not satisfy the ring rule or establish fraud.

**Technical detail:** The copilot explicitly offers household/shared-device use as an unverified hypothesis, and tests cover a legitimate shared-device case. Human verification is required.

## 12. How is the investigation queue ranked?

**Short answer:** Players are independently ranked in both domains, then the best domain rank is used so leading AML and safer-gambling cases share the review budget.

**Technical detail:** Sort key is minimum of AML rank and safer-gambling rank, followed by maximum score, secondary score, and stable entity ID. The top configured capacity—normally 5% of selected player count—is marked for review.

## 13. Why not simply rank by the maximum score?

**Short answer:** A maximum-only sort could let one domain consume the whole queue. The current policy protects capacity for both mandates.

**Technical detail:** This is a transparent prioritization heuristic, not an optimized allocation policy. It could be replaced after measuring real investigator capacity, case severity, and domain-specific costs.

## 14. How do you calculate Recall@K and Precision@K?

**Short answer:** At a budget K, Precision@K is positive cases in the top K divided by reviewed cases; Recall@K is positive cases in the top K divided by all positive cases.

**Technical detail:** After label-free ranking, `TP = reviewed intersect positive`, `FP = reviewed minus positive`, and `FN = positive minus reviewed`. Thus `Precision@K = TP/(TP+FP)` and `Recall@K = TP/(TP+FN)`. Unlinked dashboard alerts are reported separately because a player label cannot safely be assigned.

## 15. How does the Red Team arena avoid teaching evasion?

**Short answer:** It uses abstract synthetic transformations, not operational tactics.

**Technical detail:** It varies amount noise and dates, redistributes abstract channel labels, blends amounts toward a synthetic historical median, or changes observation/linkage availability. It provides no thresholds or real-world bypass procedure. The purpose is to expose detector failure under less obvious fictional data.

## 16. Is Blue Detector V2 the same algorithm as the main dashboard scorer?

**Short answer:** No. V2 is a compact benchmark used for fair Red/Blue and validation comparisons.

**Technical detail:** V2 uses the `signals.py` historical/recent features, the deterministic `models.py` weights, all observed channels, and a shared-payment signal. The main queue uses Phase 4 snapshot features, the Phase 5 anomaly/change/peer outputs, full NetworkX features, and the Phase 7 contribution-ledger scorer. Arena metrics should not be claimed as direct performance of the full queue scorer.

## 17. How is fairness handled?

**Short answer:** Age band and region are excluded from scoring and used only to audit outcomes.

**Technical detail:** Audits report group counts, flag rate, false-positive rate, false-negative rate, recall, disparity ratios, and small-sample warnings. A group is warned when it has fewer than 30 members or fewer than five positives in the rigorous pipeline. Equal flag rates do not necessarily imply fair outcomes, and synthetic results cannot establish real-world fairness.

## 18. What is the strongest technical innovation?

**Short answer:** The integration of conservative identity resolution, explicit missingness, cross-channel context, separate risk domains, explainable network evidence, and fixed-capacity human review.

**Technical detail:** None of the component algorithms is claimed as novel. The contribution is a coherent, leakage-conscious evidence architecture: uncertain histories are not merged, missing records are not zeros, ownership is not suspicious by itself, exact score contributions reconcile, labels stay outside operations, and both supporting and mitigating evidence are generated from data.

## 19. What are the most important limitations?

**Short answer:** Everything is synthetic; scores are uncalibrated; rules and weights are not validated on real outcomes; and the application is not a production case-management or data platform.

**Technical detail:** Identity evidence and scenarios are simplified, the dashboard uses seven snapshots and a reduced event count for responsiveness, the Arena uses compact benchmark detectors, and there is no production ingestion, authentication, durable audit database, model registry, drift monitoring, policy approval, or real investigator outcome feedback.

## 20. What would you build next if OLG wanted a pilot?

**Short answer:** First define a governed, privacy-preserving data contract and conduct a retrospective shadow evaluation with investigators—without automated action.

**Technical detail:** Priorities would be schema/data-quality profiling, policy-approved identity evidence, temporal backtesting, outcome definitions and adjudication, score calibration or capacity optimization, materiality floors, drift monitoring, durable role-based case workflow, security/privacy review, fairness analysis with adequate samples, and an investigator usability study. Only after those steps would model or threshold changes be justified.

## A compact answer for the live demo

> Sentinel X is not claiming to know OLG's systems or to predict guilt. It is a local synthetic prototype that demonstrates how incomplete multi-channel evidence can be integrated conservatively. It uses unsupervised ML and robust statistics to surface changes, then transparent domain-specific weights to create two review indices. Every alert retains its evidence, missingness, identity confidence, mitigating context, and human next step. Synthetic labels are held outside operations and used only to measure what the fixed investigation budget catches or misses.
