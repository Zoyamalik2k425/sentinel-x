# Sentinel X implementation plan

## Foundation audit (completed)

The starter was a working single-process Streamlit demonstration with synthetic generation, hand-built signals, dual risk indices, an investigator queue, stress testing, fixed-budget validation, and one smoke test. Its main strengths were an offline-only design, visible uncertainty, separate AML/fraud and safer-gambling scores, and computed rather than invented metrics.

The foundation audit identified these issues:

- Synthetic `scenario` and `injected_risk` labels travelled inside the profile and scored frames. Although the score formula did not reference them, this created avoidable leakage risk and exposed scenario names in Player 360.
- Model weighting and workload scoring lived in one module, so model outputs were not an explicit architectural boundary.
- Seeds, time windows, weights, and cutoffs were distributed as literals.
- The root smoke test covered only the happy path and generated thousands of timedelta deprecation warnings.
- Several UI and README characters had been decoded incorrectly.
- The documented launch command did not use the required `python -m streamlit run app.py` form.
- Scikit-learn and NetworkX were declared but not yet used. They remain available for later local modeling and graph phases; no capability is implied today.

Foundation changes separate profiles, transactions, and held synthetic labels; add configuration and model layers; add boundary and reproducibility tests; repair encoding and warnings; and document the exact local workflow.

## Target data flow

```text
Synthetic generator
  ├─ profiles ───────┐
  ├─ transactions ───┴─> feature engineering -> dual models -> risk scoring -> investigator UI
  └─ ground truth ───────────────────────────────────────────────> offline evaluation only
```

The operational branch must remain runnable without the ground-truth table. Evaluation may join predictions to labels by `player_id`, but feature engineering, model prediction, prioritization, and investigator evidence must not accept or display injected scenario labels.

## Phase 2A — realistic synthetic environment (completed)

- Added six months of reproducible event-level activity for 2,000 fictional adult players.
- Added distinct online-casino, sports-betting, and land-based schemas with valid ledger flows.
- Added correlated income, preferences, activity intensity, weekday, weekend, and time-of-day patterns.
- Added eight held scenario types, including a legitimate high-spending hard negative.
- Added nullable land-based patron linkage and incomplete wagering visibility.
- Added compressed local persistence, regeneration by seed, and structural validation tests.

No schema is presented as an OLG internal schema. Scenario labels remain outside every operational table.

## Future detection-quality phase (not started)

1. Define train/development/test seeds and time-based splits before fitting any estimator.
2. Add robust peer comparisons by channel and comparable synthetic cohorts.
3. Establish simple scikit-learn baselines, such as robust scaling plus Isolation Forest, alongside the transparent rules.
4. Add NetworkX components and shared-identifier features without replacing case-level evidence.
5. Measure recall at several fixed investigator capacities, false positives per reviewed case, scenario-level recall, and stability under missingness.
6. Tune thresholds on development data only. Report test results with uncertainty and generator assumptions.

Exit criterion: repeatable experiments whose held-out results are generated from code, with no label access in operational transformations.

## Phase 3 — cross-channel integration and identity resolution (completed)

- Added a common event contract that preserves source channel, account, event, and transaction references.
- Added confirmed, probable, unlinked, and conflicting identity states with synthetic confidence evidence.
- Restricted canonical history merging to confirmed identities; uncertain records remain source-scoped.
- Added channel coverage, missing fields, known unobserved periods, unlinked-record counts, and confidence reporting.
- Added channel and unified summaries, cross-channel change indicators, and duplicate-representation controls.
- Added a console comparison and focused tests for fragmented risk, incomplete land history, uncertain identity, single-channel activity, and identifier conflicts.

The dashboard was not changed as part of this phase.

## Future investigator workflow phase (not started)

1. Add reason codes with source records and feature-window timestamps.
2. Add queue filters for risk domain, data quality, and channel coverage.
3. Store local case notes and dispositions in an explicitly non-production persistence layer.
4. Add feedback simulation while preventing reviewed outcomes from leaking into retrospective evaluation.
5. Test accessibility, empty states, and large-queue performance.

Exit criterion: an investigator can understand why a case was prioritized, inspect supporting and counter-evidence, and record a reversible local disposition.

## Phase 4 — player behavioral feature engineering (completed)

- Added channel-specific and unified event-time feature histories.
- Added causal 7-day and 30-day activity, robust historical baselines, session changes, acceleration, and stability.
- Added cross-channel participation, combined activity, coverage, device, payment-instrument, and identity-quality context.
- Added behavioral peer comparisons without using age or region as model features.
- Retained age and region in a separate fairness-audit table.
- Added deliberate future-event leakage tests and explicit handling for incomplete history, unreliable losses, new players, and near-zero MAD.

No dashboard or model-training changes were made in this phase.

## Phase 5 — machine-learning anomaly detection (completed)

- Added an Isolation Forest pipeline with median imputation, robust scaling, deterministic fitting, and empirical 0–100 anomaly severity.
- Added causal rolling-median/MAD/CUSUM behavioral change detection.
- Added KMeans behavioral cohorts and robust training-only peer distributions without age or region.
- Added chronological train, validation, and test separation with validation F2 threshold selection.
- Added Joblib model-bundle persistence, combined inference explanations, context, and data-sufficiency status.
- Added held-out metrics and coverage-stratified comparisons using labels only after unsupervised fitting.

Anomaly indices are rankings, not calibrated fraud probabilities. Reported metrics apply only to the configured synthetic experiment.

## Phase 6 — player relationship and network intelligence (completed)

- Added NetworkX source-account graphs with typed, weighted relationship evidence.
- Added shared device, shared payment instrument, repeated one-hour activity overlap, and confirmed ownership relationships.
- Preserved probable and conflicting identities as separate account nodes.
- Added component, degree, weighted-degree, identifier-count, cluster-size, and relationship-strength metrics.
- Added conservative multi-evidence unusual-cluster screening and investigator explanations that explicitly avoid fraud determinations.
- Added deterministic Plotly-ready node and edge coordinates.
- Rejected final model scores, alerts, and scenario labels as graph inputs to prevent circular scoring.

The initial six-hour correlation window was rejected during validation because it produced a large chance-overlap component. The default is one hour with at least three repeated overlaps.

## Phase 7 — explainable risk scoring and investigation prioritization (completed)

- Added independent, configurable 0–100 AML/fraud and safer-gambling review indices.
- Added normalized weighted contribution ledgers with exact point contributions and explicit unavailable-signal status.
- Added AML transaction anomaly, limited-play fund movement, network, observed-location change, behavioral change, and cross-channel inconsistency signals.
- Added safer-gambling deposit escalation, observed-loss increase, session-frequency, gambling-intensity, and own/peer-history change signals.
- Kept data coverage and identity confidence outside both risk calculations.
- Added source transaction IDs and review-capacity selection while ranking on the stronger independent risk category.
- Rejected scenario labels and protected profile attributes at the scoring boundary.
- Kept confirmed ownership edges separate from investigative network strength so ownership alone cannot raise AML risk.

These scores prioritize synthetic cases for human review. They are not calibrated probabilities and do not authorize automated adverse action.

## Phase 8 — explainable investigator copilot (completed)

- Added deterministic, template-based cases without an LLM or external service.
- Added grounded explanations covering the flag reason, observed historical changes, evidence channels, and cited transactions.
- Separated evidence for concern from observed mitigation and explicitly labelled unverified alternative hypotheses.
- Added missing-signal, data-coverage, identity-confidence, and unresolved-transaction disclosures.
- Added human-review next steps with explicit safeguards against automated freezes, restrictions, or player contact.
- Added session-local New, Under Review, Escalated, and Closed states; investigator notes; Confirm/Dismiss controls; and timestamped audit history.
- Added narrative-grounding and workflow unit tests, including rejection of synthetic scenario labels.

This is a local prototype workflow. Its statuses and dispositions do not perform operational actions.

## Phase 9 — professional Streamlit investigator dashboard (completed)

- Replaced the legacy four-tab demonstration with five focused workspaces: Overview, Investigation Queue, Player 360, Network Intelligence, and Validation & Fairness.
- Added a dark navy financial-intelligence visual system, responsive metrics, risk charts, ranked tables, search, and operational filters.
- Connected the interface to the Phase 3–8 pipeline, including separate risk ledgers, cited evidence, local case workflow, and complete graph metrics.
- Added interactive account-network visualization with evidence-ranked layout capping; capping affects only rendered nodes, never graph analysis or scoring.
- Added computed Recall@K, Precision@K, false-positive/negative counts, per-scenario and per-channel performance, and age/region audit views.
- Preserved the visibility stress benchmark inside validation and kept held labels outside operational scoring.
- Added cached causal feature snapshots for large local populations and tests that future transactions cannot alter fixed earlier snapshots.
- Added navigation and page-rendering regression tests for all five workspaces.

All figures are computed from the current fictional run. No performance result is hard-coded.

## Phase 10 — defensive Red vs Blue arena (completed)

- Added four abstract behavioral-adaptation levels and four channel/identity visibility conditions.
- Compared simple rules, Blue V1, and Blue V2 on separate development and held-out populations with identical budgets.
- Added scenario/coverage recall, false-positive/negative, workload, and model-failure views.
- Added the sixth Streamlit workspace and deterministic experiment caching.

## Phase 11 — model validation and fairness auditing (completed)

- Added separate AML/fraud and safer-gambling threshold and fixed-budget metrics.
- Added comparable baseline, single-channel, cross-channel, and cross-channel-plus-network score paths.
- Added scenario/channel recall and sensitivity for visibility loss, identity uncertainty, cold start, benign high spending, and geographic change.
- Added age/region flag, false-positive, false-negative, recall, disparity-ratio, and small-sample reporting.
- Preserved protected attributes as evaluation-only fields and documented that equal flag rates do not establish fairness.

## Phase 12 — final integration and presentation (completed)

- Repaired the dashboard preparation/cache boundary and expanded page regression coverage.
- Reviewed label, identity, missing-data, scoring, explanation, and network trust boundaries.
- Added architecture, model card, computed evaluation report, five-minute demo, and limitations documentation.
- Consolidated setup, verification, architecture, scope, and handoff guidance in the README.

## Next recommended increment

Run accessibility and security reviews, calibrate thresholds on representative authorized development data, add uncertainty intervals across multiple temporal seeds, and design durable role-based case management. Preserve human decision authority and do not add automatic interventions.
