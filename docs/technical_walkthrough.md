# Sentinel X technical walkthrough

## Read this first

Sentinel X is a local, synthetic-data investigator decision-support prototype. It does not use OLG data, does not describe OLG's internal schemas or controls, and does not make automated account decisions. Its outputs are review indices and investigation leads, not calibrated probabilities of fraud or gambling harm.

This guide starts with a plain-language explanation and then gives the implementation detail needed to defend the design to a computer-science audience.

## 1. The application in one minute

In simple terms, Sentinel X does this:

1. Creates fictional players and fictional activity in an online casino, sportsbook, and land-based casino.
2. Converts the three different event formats into one common format.
3. Links source accounts only when the synthetic identity evidence is strong enough. Uncertain accounts stay separate.
4. Builds time-aware behavioral features, such as deposits in the last 7 and 30 days.
5. Runs an unsupervised anomaly model, a statistical change detector, and behavioral peer analysis.
6. Builds a relationship graph from shared synthetic identifiers, overlapping activity, and confirmed ownership.
7. Combines those signals into separate AML/Fraud and Safer Gambling review indices.
8. Ranks a fixed-size human review queue and creates evidence-based case narratives.
9. Uses the held-back synthetic scenario labels only afterward, in validation and defensive stress testing.

The dashboard's operational path is:

```text
data_gen.generate_dataset
        |
        v
cross_channel.integrate_channels
        |
        v
dashboard_features.build_dashboard_feature_history
        |
        +--> ml.inference.run_inference
        |
        +--> network_analysis.build_relationship_graph
        |
        v
scoring.score_risk_indices
        |
        +--> scoring.prioritize_investigations
        |
        +--> investigator_copilot.build_investigation_case
        |
        v
app.py pages and charts
```

`scenario_labels` is not passed through that path. It is read separately by validation screens and experiments after scores have been produced.

## 2. What every Python file does

### Application and configuration

- `app.py` is the Streamlit entry point. It assembles the full pipeline into a `DashboardData` object, caches it, applies the visual theme, and renders Overview, Investigation Queue, Player 360, Network Intelligence, Validation & Fairness, and Red vs Blue Arena views. The default is 2,000 players; the interactive event target is currently 10 events per player, and the review budget is controlled in the UI. If no saved model bundle exists, it trains a local fallback bundle from the generated history.
- `configuration.py` holds reproducible defaults. `DataConfig` defines 2,000 players, 182 days, a 100,000-event standalone generation target, seed 42, and a 3% injected positive-scenario fraction. `FeatureConfig` and `ModelConfig` contain settings for the older validation/baseline path.

### Synthetic data and integration

- `data_gen.py` creates fictional profiles and three channel-specific event tables. It injects synthetic scenarios, maintains per-channel running balances, validates invariants, saves compressed CSV files, and exposes the named `OperationalDataset` contract used by defensive evaluation. Labels are returned in their own DataFrame.
- `cross_channel.py` creates fictional identity evidence, resolves confirmed/probable/unlinked/conflicting accounts, normalizes channel schemas, removes duplicate source representations, and calculates summaries and completeness indicators. Probable and conflicting identities are not folded into a confirmed player's history.
- `demo_cross_channel.py` is a small console demonstration of single-channel versus unified observations. It exercises operational integration without reading scenario labels.

### Features and model signals

- `feature_engineering.py` is the full event-level Phase 4 implementation. Every feature row is computed from records at or before that event time. It produces channel-only features, unified features, coverage fields, and causal behavioral-peer comparisons. Age and region are exposed only through the separate `fairness_audit_attributes` function.
- `dashboard_features.py` is the faster interactive equivalent. Rather than create a row for every event, it creates seven causal historical snapshots by default. This preserves the model-facing concepts while making a 2,000-player dashboard practical. The full event-level builder remains the training/evaluation reference implementation.
- `ml/anomaly_detection.py` wraps scikit-learn's `IsolationForest` in median imputation and robust scaling. It maps raw anomaly scores to a 0-100 empirical percentile of the training-score distribution.
- `ml/behavioral_change.py` implements a within-player rolling median/MAD and positive CUSUM-style change score. Same-timestamp rows share the same prior context, which prevents one event from leaking into another at the same timestamp.
- `ml/peer_analysis.py` uses KMeans to form behavioral peer clusters, then measures robust deviation from the training-only cluster distribution. Age and region are explicitly forbidden as grouping or comparison inputs.
- `ml/train.py` performs chronological 60%/20%/20% train/validation/test separation, fits the unsupervised models without labels, selects an alert threshold on validation labels using F2, saves the bundle with Joblib, and reports held-out metrics. Labels are used for threshold evaluation, not model fitting.
- `ml/inference.py` loads/saves the model bundle, runs the three model components, reports their outputs and top indicators, and assigns a data-sufficiency status. Its `model_priority_index` is the maximum of anomaly severity, behavioral-change score, and peer-deviation score; this is an intermediate model output, not the final queue score.
- `ml/__init__.py` marks `ml` as a Python package and provides the package boundary for these modules.

### Explainability, relationships, and risk

- `network_analysis.py` creates a NetworkX graph whose nodes are source accounts. Typed edges represent shared devices, shared payment tokens, repeated overlapping activity, or confirmed cross-channel ownership. It calculates components and account metrics, flags multi-evidence clusters, creates deterministic relationship explanations, and prepares coordinates for Plotly.
- `scoring.py` contains the current Phase 7 risk scorer, exact contribution ledgers, risk bands, supporting transaction IDs, and capacity-aware queue ranking. It also retains an older `score`/`explain` API for the lightweight validation detectors.
- `investigator_copilot.py` builds deterministic case records: why flagged, historical changes, evidence for concern, observed mitigation, explicitly unverified alternative hypotheses, missing information, cited transactions, and recommended human-review steps. It also implements the local status/notes/audit workflow in Streamlit session state.
- `dashboard_metrics.py` turns scored results into friendly queue rows and calculates dashboard validation, scenario, channel, and fairness tables. Held labels enter only these evaluation functions, after the queue exists.

### Validation and adversarial testing

- `signals.py` implements a compact, label-free historical/recent feature set used by the validation and Red Team arena detectors. It is deliberately simpler than the main Phase 4/5/7 pipeline.
- `models.py` implements transparent weighted heuristic indices for that compact validation path. It is not a fitted estimator.
- `validate.py` contains two evaluation layers: the three-detector Arena comparison and a broader four-detector, two-risk-domain validation pipeline. It calculates fixed-threshold metrics, fixed-budget metrics, scenario/channel recall, missing-data sensitivity, cold-start and benign-control cohorts, and fairness audit tables.
- `red_team.py` applies predefined, abstract transformations to fictional observations and reruns the same detectors, population, and alert budget. It keeps development and held-out evaluation seeds separate.

### Automated tests

- `tests/test_data_gen.py` checks identifiers, timestamps, balances, reproducibility, saved data, and label separation.
- `tests/test_cross_channel.py` checks cross-channel uplift, missing land observations, uncertain/conflicting identities, one-channel players, and deduplication.
- `tests/test_feature_engineering.py` checks causal features, future-data leakage, missing land losses, cold starts, near-zero MAD, peer design, and label rejection.
- `tests/test_dashboard_features.py` checks the faster snapshots against the model contract and deliberately verifies that future records cannot alter earlier snapshots.
- `tests/test_ml.py` checks missing-value handling, change-score causality, peer features, chronological splits, leakage rejection, model persistence, inference, and evaluation.
- `tests/test_network_analysis.py` checks unrelated accounts, legitimate sharing, multi-evidence clusters, uncertain identity, ownership edges, forbidden score inputs, and Plotly data.
- `tests/test_scoring.py` checks score bounds, contribution reconciliation, explicit missingness, category independence, supporting IDs, demographic/label rejection, ranking, confidence separation, and graded signals.
- `tests/test_investigator_copilot.py` checks that narratives match supplied evidence, uncertainties are disclosed, hypotheses remain unverified, labels are rejected, queue selection is respected, workflow history works, and currency escaping happens only at rendering.
- `tests/test_dashboard_metrics.py` checks queue projections and computed fixed-budget, scenario, channel, and fairness results.
- `tests/test_red_team_arena.py` checks repeatability, non-mutation, visibility conditions, separate populations, fixed workload, and distinct seeds.
- `tests/test_validation_pipeline.py` checks label/demographic separation, independent risk domains, common population and budget, sensitivity/fairness outputs, and plots built from computed tables.
- `tests/test_pipeline.py` is a broad integration suite for reproducibility, dual outputs, held-label evaluation, leakage rejection, and stress-level validation.
- `tests/test_app_integration.py` protects the named generator contract, complete dashboard payload, and Streamlit startup path.

## 3. How synthetic player data is generated

### Simple explanation

The generator creates adults represented by age band, region, account date, income band, occupation category, gaming preference, and account status. It then gives each fictional player a latent weekly budget and activity level. Those hidden behavioral quantities influence how often and how much the player transacts, so activity is correlated rather than a collection of unrelated random numbers.

All random draws come from `numpy.random.default_rng(seed)`. Reusing the same arguments and seed reproduces the same profiles, events, and labels.

### Technical explanation

The default standalone configuration is 2,000 players over 182 days and 100,000 events. `app.py` intentionally uses a lighter interactive target of `10 * player_count`, so its default 2,000-player dashboard creates 20,000 events before caching. The standalone generator divides its requested total approximately 46% online casino, 34% sportsbook, and 20% land casino.

For each player:

- Annual-income band is mapped to a synthetic midpoint.
- Weekly budget is `clip(annual_income / 52 * LogNormal(-2.75, 0.52), 12, 1500)`.
- A separate `LogNormal(0, 0.72)` activity factor controls event counts.
- Channel preference multiplies activity by 2.2 for a preferred channel and 0.42 otherwise.
- Event dates use weighted daily sampling. Weekends are more likely, especially for land-based play; event hours are drawn around channel-specific evening means.

Channel schemas remain distinct:

- Online casino has deposits, wagers, payouts, withdrawals, sessions, game category, devices, payment instruments, region, and before/after balance.
- Sportsbook has deposits, wagers, bet IDs/timestamps, settlements, withdrawals, sport category, devices, payment instruments, and balance.
- Land casino has buy-ins, cash-outs, optional wager/payout observations, optional patron/loyalty linkage, property region, and potentially missing balances or wagering observations.

Balances are updated event by event. Casino balance change is `deposit + payout - withdrawal - wager`; sportsbook uses `deposit + settlement - withdrawal - wager`; linked land activity uses `buy_in + payout - cash_out - observed_wager`. Amounts that would overdraw a balance are capped, and the validation function tests ledger equality and non-negative balances.

The generator assigns approximately 3% of players to positive scenarios, spread round-robin across observed age-band/region strata. It also creates about 0.5% legitimate high-spender negative controls. Current positive scenario names include structuring, income mismatch, mule ring, sudden spending escalation, increasing losses/deposits, unusual geographic activity, coordinated relationships, rapid funds/limited play, distributed cross-channel activity, and abrupt change. These are fictional test constructs, not behavioral definitions supplied by OLG.

Scenario injection affects only synthetic event creation. For example, it can increase post-start event frequency/amounts, change the mix of deposits/wagers/withdrawals, change observed regions, or create shared synthetic identifiers. `scenario_labels` contains only `player_id`, scenario type, start/end, target risk category, and Boolean label. Those fields are absent from profiles and operational events, and feature/model/scoring boundaries reject them if accidentally supplied.

## 4. How different channels are combined

### Simple explanation

Every source event is translated into a shared vocabulary: who or which source account it belongs to, when it happened, what kind of money movement it represents, what was observed, and how confident the identity link is. The original channel and source event ID are retained.

The key safety rule is: uncertainty is preserved. A probable land-based match is not silently added to a confirmed online history.

### Technical explanation

`cross_channel.py` first builds source-account evidence:

- Authenticated online-casino and sportsbook accounts receive synthetic confidence 1.0.
- Land records may use a loyalty ID, patron linkage, or an event-specific unlinked account ID.
- Identity resolution classifies each `(channel, source_account_id)` as confirmed, probable, unlinked, or conflicting.
- Confirmed requires one valid asserted player and confidence at least 0.95. Probable is at least 0.60 but below 0.95. Multiple asserted players or an explicit conflict produces `conflicting`.

Only confirmed identities get `activity_entity_id = PLAYER:<player_id>`. Every other account gets a separate `SOURCE:<channel>:<source_account_id>` entity, even if it has a proposed player ID.

Normalization maps deposits/buy-ins, withdrawals/cash-outs, wagers, and payouts/settlements into common amount columns. Land wagering/payout fields become `NaN` when the source says that observation was unavailable. This is how the system distinguishes unknown activity from observed zero activity.

Deduplication first removes repeated `(channel, source_event_id)` rows. It then removes duplicate representations with the same transaction reference, financial direction, and rounded amount, keeping the row with the best observed/confidence/completeness information. It does not collapse different transfer legs merely because amounts match.

The integration result includes:

- channel-specific summaries;
- unified confirmed-entity summaries;
- identity and data-completeness information;
- observed channel coverage and unobserved periods;
- unified versus largest-channel deposit change indicators.

## 5. Features and temporal leakage protection

### Simple explanation

For an event at time T, the feature builder is allowed to look backward to T, never forward past T. It calculates recent activity, a longer baseline, changes from that baseline, and how complete the observation is.

### Technical explanation

The full builder groups records both by `(activity_entity_id, channel)` and by `activity_entity_id`. For each timestamp-bounded row it calculates 7-day and 30-day deposit frequency/amount, withdrawal frequency, wagering volume, reliable net loss, session frequency, median deposit, median absolute deviation (MAD), stability, devices, payment instruments, time since prior event, observation span, active channels, participation change, and coverage.

Important formulas include:

- Spending acceleration = `(deposit_7d / 7) / (deposit_previous_23d / 23) - 1` when the prior window is observed and positive.
- Session-frequency change uses the same 7-day versus previous-23-day rate formula.
- Behavioral stability = `1 / (1 + MAD / max(|median|, 1))` with at least three historical deposits.
- Robust peer z-score = `(player_30d_deposits - peer_median) / (1.4826 * peer_MAD)` when at least five peers exist and MAD is nonzero.
- Channel coverage = `observed_active_channels / 3`.

Behavioral peer groups begin from gaming preference, not age or region. The learned Phase 5 peer analyzer independently clusters historical behavioral variables with KMeans. Age and region are kept in a separate audit table.

The tests append deliberately future-dated transactions and assert that earlier feature rows remain byte-for-byte equivalent on the tested columns.

## 6. What is machine learning, statistical, and heuristic

Sentinel X uses all three, and they should not be conflated.

### Trained machine-learning components

1. **Isolation Forest** is a trained unsupervised scikit-learn model. Its pipeline is median imputation with missing indicators, `RobustScaler`, and 250-tree `IsolationForest(contamination=0.08, random_state=seed)`. It does not train on scenario labels. Severity is the empirical percentile of `-decision_function` relative to sorted training scores:

   `anomaly_severity = 100 * count(training_raw <= current_raw) / number_of_training_rows`

   This is a rank-like anomaly index, not a fraud probability.

2. **Behavioral peer analysis** is trained unsupervised. KMeans uses up to six clusters on historical behavioral features. Within each cluster, current values are compared with training medians and robust scales. The largest absolute robust deviation `d` becomes:

   `peer_deviation_score = 100 * (1 - exp(-d / 3))`

### Statistical component

The behavioral-change detector is not a learned predictive model. For each player it takes strictly earlier values, calculates median and MAD, and uses scale:

`scale = max(1.4826 * MAD, 0.10 * |median|, 1)`

It calculates a positive robust z-score, updates a one-sided CUSUM with drift 0.5, takes the larger of immediate change and normalized accumulated change, and maps strength `s` to:

`behavioral_change_score = 100 * (1 - exp(-s / 3))`

At least five prior observations are required; otherwise the result is missing, not zero.

### Heuristic components

The final dual risk indices, network edge strengths, cluster flag rule, queue policy, narrative threshold, simple baseline rules, and Red/Blue detector weights are transparent heuristics. The risk weights were chosen as configurable prototype values; they were not learned from real-world outcomes.

The dashboard usually uses the trained Phase 5 components plus the heuristic Phase 7 scorer. If `artifacts/phase5_models.joblib` is absent, it trains a local fallback from the current synthetic feature history. The validation/Arena path uses the smaller `signals.py` features and `models.py` weighted rules, not the full dashboard scorer. Therefore an Arena result should not be presented as direct validation of the exact Phase 7 queue algorithm.

## 7. Exact AML/Fraud and Safer Gambling score calculation

### Step A: raw operational signals

AML/Fraud raw signals are:

- `transaction_anomaly = anomaly_severity / 100`.
- `unusual_fund_movement` is the maximum of two recent 30-day patterns: withdrawal-to-deposit ratio with limited wagering, and repeated deposits above three events scaled over the next nine events, again discounted by play. It is missing if deposits/withdrawals are not observable.
- `network_relationships` is the maximum incident investigation strength from shared identifier/correlated-activity edges. Confirmed ownership alone contributes zero investigation strength.
- `geographic_anomaly` is the fraction of observed locations in the latest seven days that differ from the entity's modal location before that window.
- `behavioral_change = behavioral_change_score / 100`.
- `cross_channel_inconsistency = 0.7 * deposit_gap * multi_channel_factor + 0.3 * participation_change_factor`, where `deposit_gap = |unified_30d - largest_channel_30d| / unified_30d`, `multi_channel_factor = clip((active_channels - 1)/2)`, and participation change is capped after division by two.

Safer Gambling raw signals are:

- `deposit_escalation = clip(max(spending_acceleration, 0) / 3)`.
- `observed_net_loss_increase` compares the seven-day share of reliable 30-day loss with the stable-rate expectation `7/30`: `clip((loss7/loss30 - 7/30) / (1 - 7/30))`.
- `session_frequency_increase = clip(max(session_frequency_change, 0) / 3)`.
- `gambling_intensity_change` applies the same seven-day-share formula to wager volume.
- `historical_pattern_change = clip(|peer_deposit_robust_z| / 5)`.

`clip` means cap to the interval 0 to 1.

### Step B: population grading

Each available raw signal is graded against the current scored population:

`graded_signal = min(0.98, 0.70 * raw_signal + 0.30 * percentile_rank)`

A true raw zero remains zero. Missing remains missing. The 0.98 cap prevents every strong signal from displaying as 100%.

### Step C: weighted indices

AML/Fraud configured weights are:

| Signal | Weight |
|---|---:|
| Transaction anomaly | 0.40 |
| Unusual fund movement | 0.30 |
| Network relationships | 0.20 |
| Geographic anomaly | 0.04 |
| Behavioral change | 0.03 |
| Cross-channel inconsistency | 0.03 |

Safer Gambling configured weights are:

| Signal | Weight |
|---|---:|
| Deposit escalation | 0.25 |
| Reliably observed net-loss increase | 0.25 |
| Session-frequency increase | 0.15 |
| Gambling-intensity change | 0.20 |
| Historical-pattern change | 0.15 |

Missing signals are not treated as zero. For each domain, weights are renormalized across only available signals:

`normalized_weight_i = configured_weight_i / sum(configured weights of available signals)`

`contribution_i = 100 * graded_signal_i * normalized_weight_i`

`risk_index = clip(sum(contribution_i), 0, 100)`

The ledger separately reports `evidence_weight_coverage = available configured weight / total configured weight`. This means a high score based on limited evidence can remain high, but the investigator is explicitly warned that coverage is limited. Identity confidence and data coverage do not multiply the risk score.

Risk bands use the maximum of the two independent indices: low below 35, moderate from 35, high from 60, and critical from 80.

### The older validation formula

The compact validation/Arena model in `models.py` is different:

- AML/Fraud = `100 * (0.30*change + 0.40*network + 0.20*geo + 0.10*cross_channel)`.
- Safer Gambling = `100 * (0.50*change + 0.50*loss)`.

The Simple Baseline uses the maximum of its compact change/geographic signals for AML and change/loss signals for safer gambling. These are benchmark rules, not the dashboard's exact contribution-ledger formula.

## 8. How the relationship graph works

Each graph node is a source account, such as one sportsbook account or one land loyalty account. Edges are added for:

- one or more shared device IDs;
- one or more shared payment-instrument tokens;
- activity in at least three of the same one-hour windows;
- confirmed cross-channel ownership by the same fictional player.

Very large identifier groups and time windows are ignored by configurable limits to avoid creating meaningless all-to-all links.

Component strengths are combined without simple addition. For device component `D`, payment component `P`, correlation component `C`, and ownership component `O`:

- `D = min(0.45, 0.20 * shared_devices)`
- `P = min(0.55, 0.25 * shared_payment_tokens)`
- `C = min(0.70, 0.12 * overlapping_windows)`
- `O = 0.90` for confirmed ownership, otherwise 0
- `relationship_strength = 1 - (1-D)(1-P)(1-C)(1-O)`
- `investigation_strength = 1 - (1-D)(1-P)(1-C)`

Ownership is deliberately excluded from investigation strength. Normal ownership can join a player's source accounts without becoming AML evidence.

A component is marked `Ring flagged` only when it has at least three accounts, at least two shared-identifier edges, and either at least two correlated-activity edges or strong multi-identifier evidence across at least three edges. This is a transparent heuristic lead, not a fraud classification. Every edge explanation ends by saying that it is an investigation lead, not proof.

The full graph supplies scoring statistics. The visual layout may be capped to 260 prioritized nodes for responsiveness, using a seeded NetworkX spring layout; that cap does not change full graph metrics.

## 9. How the investigation queue is ranked

The queue does not simply sort by the larger score. It intentionally gives both independent risk domains access to review capacity:

1. Rank every entity descending by AML/Fraud score.
2. Independently rank every entity descending by Safer Gambling score.
3. Set `domain_rank = min(AML rank, Safer Gambling rank)`.
4. Sort ascending by `domain_rank`.
5. Break ties by the larger of the two scores, then the smaller score, then stable entity ID.
6. Assign `priority_rank = 1, 2, ...`; select ranks up to the review budget.

This interleaves leading AML and safer-gambling cases instead of letting one domain crowd out the other. The default UI budget is 5% of the chosen player count. An unlinked source account can appear, but it retains a friendly source-account name and is not presented as a confirmed player.

## 10. How evidence for and against concern is generated

The copilot is deterministic template code; no LLM or external API is used.

Evidence for concern is produced from a signal only when its graded value is at least 0.30. The text includes the measured percentage and exact score contribution. Supporting transactions must actually exist in the supplied event table and are selected from the scorer's recent high-amount IDs.

Observed mitigating evidence is data-derived and may include:

- at least 365 days of recorded account tenure;
- behavioral stability of at least 70%;
- deposits no higher than the 75th behavioral-peer percentile;
- no supplied shared-device/payment infrastructure signal;
- observed weekend or Ontario public-holiday overlap.

Alternative explanations are always labelled `Unverified hypothesis`. They are selected only when relevant signals are active—for example, ordinary wallet management for funds movement, household sharing for network evidence, travel/data limitations for geography, or temporary discretionary change for increased gambling activity. The system never states these as facts.

Missing evidence lists unavailable signals, incomplete channel coverage, explicitly unobserved records, limited history, uncertain identity, and unresolved supporting IDs. Every case gets at least the recommendation to record the human disposition and take no automatic account, restriction, or player-contact action. Additional recommendations depend on cited records, identity uncertainty, network evidence, and whether either index is at least 35.

Statuses are New, Under Review, Escalated, and Closed. Notes, confirmations/dismissals, and audit history are stored locally in Streamlit session state; this is not a durable multi-user case-management system.

## 11. Red Team vs Blue Team experiments

The Arena is a defensive synthetic stress test. It does not provide instructions for evading monitoring.

Experiment A applies four predefined levels to the same synthetic positive cases:

- Level 0 leaves obvious injected observations unchanged.
- Level 1 multiplies positive-case amounts by seeded factors drawn around 1.0, capped from 0.65 to 1.35, and shifts timestamps up to two days.
- Level 2 also spreads timestamps up to four more days and redistributes abstract channel labels.
- Level 3 blends positive amounts toward each player's pre-scenario median: `0.35 * transformed_amount + 0.65 * historical_median`.

Experiment B changes observation only:

- Single-channel marks non-online-casino records unobserved.
- Complete cross-channel leaves observations unchanged.
- Partial cross-channel hides a seeded 45% of non-online observations.
- Uncertain linkage reassigns a seeded 30% of sportsbook/land event player IDs to event-specific unlinked IDs.

Every condition evaluates the same population and fixed alert budget. The three detectors are:

- Simple Baseline Rules: online-casino-only, no shared-payment signal.
- Blue Detector V1: all observed channels, no shared-payment signal.
- Blue Detector V2: all observed channels plus the compact shared-payment-token signal.

Development and held-out populations use different seeds. Labels identify which fictional positive events receive stress transformations and evaluate outcomes, but never enter detector features. Results are measured at runtime; no improvement is hardcoded or guaranteed.

## 12. Recall@K and Precision@K

Let K be the investigation budget. Sort the label-free score table descending and review the first `min(K, population size)` unique players.

- True positives (TP): reviewed players whose held label is positive.
- False positives (FP): reviewed players without a positive held label.
- False negatives (FN): positive held-label players outside the reviewed set.
- `Precision@K = TP / (TP + FP)`, which is also `TP / number reviewed` here.
- `Recall@K = TP / (TP + FN)`, which is also `TP / total positive players`.

Example: if K=50, 50 players are reviewed, 12 are positive, and the population contains 15 positives, then Precision@50 is `12/50 = 24%` and Recall@50 is `12/15 = 80%`.

The dashboard queue evaluation excludes unlinked alerts from player-label precision because they cannot safely be matched to a held player label, and reports their count separately. The broader validation pipeline evaluates AML/Fraud and Safer Gambling positives separately based on each scenario's target category.

Equal flag rates across demographic groups are not proof of fairness. The audit reports group size, rates, errors, recall, disparity ratios, and small-sample warnings. Age and region are audit attributes only, not scoring inputs.

## 13. Complete worked example: Player P000291

This trace was computed from the actual code with seed 42, 300 players, 3,000 requested events, 182 days, and a 15-account review budget. A smaller population is used here so the example is easy to reproduce. The player was selected as the first confirmed identity in the operational queue without reading `scenario_labels`.

### Generated profile and activity

P000291 is a fictional active account in the 45-59 age band, East region, opened 2024-03-28, with synthetic `40k_70k` income band, professional occupation, and casino preference. Age, region, income, and occupation did not enter scoring.

Identity resolution merged authenticated online and sportsbook accounts into `PLAYER:P000291` with confirmed status and confidence 1.0. The normalized history had 19 online-casino and 4 sportsbook records. Observed totals were:

| Channel | Deposits | Withdrawals | Wagers | Payouts/settlements | Net loss |
|---|---:|---:|---:|---:|---:|
| Online casino | \$45.48 | \$0.00 | \$36.92 | \$17.25 | \$19.67 |
| Sportsbook | \$15.00 | \$0.00 | \$3.44 | \$0.00 | \$3.44 |

At the final causal snapshot, the relevant recent features were:

- two deposits totalling \$10 in seven days and \$15 in 30 days;
- spending-rate acceleration `5.5714`, meaning +557% versus the preceding 23-day daily rate;
- session-rate change `5.5714`, also +557%;
- one active observed channel in the latest 30 days, so coverage `1/3 = 33.3%`;
- 30-day net loss unavailable because the required loss observations did not meet the 95% reliability rule;
- behavioral-peer deposit percentile 77.8%, but peer robust z-score unavailable;
- 825 days of account tenure and historical spending stability 1.0.

### Model outputs

The fitted local models produced:

- anomaly severity 98.778;
- behavioral-change score unavailable because sufficient comparable history was not available;
- peer-deviation score 63.212;
- intermediate model priority 98.778, the maximum of those available outputs;
- data sufficiency `limited`.

The anomaly explanation's leading standardized indicators included spending acceleration 10.16 IQR-like units, 30-day deposit frequency 3.00, 7-day deposit frequency 2.00, and 30-day session frequency 3.00. These are diagnostic deviations, not score contributions.

### Exact final risk ledgers

For AML/Fraud, transaction anomaly graded to 0.98. Fund movement, network, geography, and cross-channel inconsistency were observed zeros; behavioral change was missing. Available AML configured weight was 0.97, so transaction anomaly's normalized weight was `0.40 / 0.97 = 0.412371`. Its contribution was:

`100 * 0.98 * 0.412371 = 40.4124 points`

All other available AML contributions were zero. Therefore AML/Fraud index = **40.41**, with 97% evidence-weight coverage.

For Safer Gambling, deposit escalation and session increase both graded to 0.98. Net-loss increase, gambling-intensity change, and historical peer change were missing. Available safer-gambling weight was `0.25 + 0.15 = 0.40`.

- Deposit contribution = `100 * 0.98 * (0.25/0.40) = 61.25`.
- Session contribution = `100 * 0.98 * (0.15/0.40) = 36.75`.
- Safer Gambling index = `61.25 + 36.75 = 98.00`.

Evidence-weight coverage was only 40%. The 98.00 result therefore means “very strong among available safer-gambling signals,” not “98% probability” and not “complete evidence.” The missingness warning remains attached.

### Priority and case explanation

The entity ranked first under the balanced domain-rank queue, was selected within the 15-account budget, and received a critical band because its larger index was at least 80. The primary domain was Safer Gambling.

The cited source records were three real rows from the generated table: online-casino deposits `OC000001332` and `OC000001331` for \$5 each in the latest seven days, plus earlier deposit `OC000001330` for \$5.

Evidence for concern stated the exact three nonzero narrative signals and contributions: 98% transaction anomaly/40.41 AML points, 98% deposit escalation/61.25 safer-gambling points, and 98% session increase/36.75 safer-gambling points.

Evidence against or alternative context stated that the account had 825 days of recorded tenure, stability measured 100%, no shared infrastructure was found, and 10 of 23 supplied events overlapped weekends or Ontario public-holiday dates. It then labelled two possibilities as unverified hypotheses: ordinary wallet management/withdrawal of winnings, and a temporary discretionary-activity change.

The case explicitly disclosed unavailable behavioral-change, reliable net-loss, gambling-intensity, and peer-deviation signals; 33% current channel coverage; and limited history. Recommended steps were to verify the cited ledger sequence, route the observations to authorized safer-gambling and AML reviewers, record the investigator's reasoning, and take no automated adverse action.

This example also demonstrates an honest limitation: small absolute amounts can rank highly in a small synthetic population because the model detects relative acceleration and population anomaly. A real deployment would need policy-calibrated materiality, real outcome studies, and substantially more representative data.

## 14. What is genuinely innovative here

The individual algorithms—Isolation Forest, KMeans, MAD, CUSUM, weighted scores, and graphs—are established techniques. The innovation is their integration around the sponsor problem:

- heterogeneous channel schemas are retained while producing one analysis layer;
- uncertain identities remain separate instead of being optimistically merged;
- missing land-based observations remain unknown rather than zero;
- channel-only and unified context can reveal fragmented behavior;
- AML/Fraud and Safer Gambling remain separate throughout scoring and queue display;
- risk strength, identity confidence, and data completeness are separate quantities;
- network ownership links are separated from investigative network strength;
- every score has an exact contribution ledger and cited supporting records;
- the two-column narrative includes both concern and data-derived mitigation;
- a fixed-capacity, recall-oriented evaluation makes investigator workload explicit;
- adversarial visibility experiments expose failures as well as successes;
- protected demographic fields are reserved for outcome auditing.

The strongest hackathon claim is therefore not “a new anomaly algorithm.” It is “a transparent, leakage-conscious decision-support workflow for cross-channel evidence under unequal data quality.”

## 15. Technical limitations and unimplemented capabilities

- All people, schemas, relationships, outcomes, and results are synthetic. External validity is unknown.
- The project has no access to OLG systems and makes no claim about their architecture, identifiers, thresholds, or data quality.
- Scenario injection is simplified and can make cases easier to detect than real behavior.
- Main scoring weights, risk bands, narrative thresholds, graph strengths, and ring rules are hand-configured, not learned or policy-approved.
- Anomaly severity and both risk indices are not calibrated probabilities.
- Validation labels are synthetic injections, not adjudicated investigator outcomes.
- The Arena benchmarks compact heuristic detectors, not the exact full Phase 7 scorer.
- The local fallback model is fitted on the current synthetic history; this is demonstration behavior, not production model governance.
- The dashboard snapshot path has seven historical cutoffs rather than one feature row per event.
- The interactive app currently generates 20,000 events at the 2,000-player default, below the standalone Phase 2 generator's 80,000-120,000 demonstration target. This is a responsiveness trade-off.
- Identity evidence is synthetic. There is no real probabilistic entity-resolution model, clerical-review feedback loop, or survivorship handling.
- Geographic change uses observed region consistency; there is no IP intelligence, travel verification, or geospatial model.
- Shared devices/payment tokens are only leads. Household, business, public-network, and device-recycling context is not modeled.
- Net loss is withheld when observation reliability is inadequate, but the prototype cannot recover unobserved activity.
- The queue policy interleaves domains but has not been optimized with investigator service times or harm costs.
- Streamlit session state is not durable, authenticated, concurrent, or auditable enough for production case management.
- There is no production ingestion, schema registry, message queue, data lineage service, access control, encryption/key management, retention policy, drift monitor, model registry, approval workflow, or incident response integration.
- Fairness reporting is descriptive and often sample-limited. Equal rates do not prove fairness, and synthetic demographic distribution cannot establish real-world equity.
- There is no causal inference, supervised outcome model, calibrated uncertainty interval, counterfactual explanation, or human-factor usability study.
- The system recommends human review only. It does not freeze accounts, restrict gambling, contact players, file reports, or make legal/compliance decisions.

## Presentation-safe summary

Sentinel X currently combines trained unsupervised ML, robust statistical change detection, and transparent heuristic scoring. It is technically real and locally executable, but it remains a synthetic investigator decision-support prototype. Present the computed evidence and failure modes confidently; do not present the indices as probabilities, the labels as real outcomes, the Arena as proof of production performance, or the schemas as OLG's.
