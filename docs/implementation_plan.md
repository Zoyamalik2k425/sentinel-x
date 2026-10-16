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

## Phase 3 — investigator workflow

1. Add reason codes with source records and feature-window timestamps.
2. Add queue filters for risk domain, data quality, and channel coverage.
3. Store local case notes and dispositions in an explicitly non-production persistence layer.
4. Add feedback simulation while preventing reviewed outcomes from leaking into retrospective evaluation.
5. Test accessibility, empty states, and large-queue performance.

Exit criterion: an investigator can understand why a case was prioritized, inspect supporting and counter-evidence, and record a reversible local disposition.

## Phase 4 — governance and demonstration readiness

1. Create model cards and a synthetic-data datasheet.
2. Add schema validation, structured logging without sensitive values, and dependency/security checks.
3. Document failure modes, escalation boundaries, retention assumptions, and human-oversight requirements.
4. Prepare a reproducible demo script that clearly distinguishes synthetic results from real-world claims.

Exit criterion: another team member can reproduce the demo and explain its limits without implying knowledge of OLG's internal architecture.

## Next recommended increment

Build the experiment harness first: introduce explicit development/test seeds, a time-aware evaluation split, scenario-level recall, and a transparent scikit-learn baseline. Keep the existing rules as the benchmark and do not change UI claims until held-out results have actually been computed.
