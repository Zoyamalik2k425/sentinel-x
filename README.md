# Sentinel X — ThunderHacks 2026

Sentinel X is a local, explainable investigator decision-support prototype for cross-channel player protection and fraud review. Every player, event, identity signal, scenario, and reported metric is synthetic. The project does not use or describe OLG internal data, databases, controls, or architecture.

The prototype detects unusual activity, adds historical and peer context, prioritizes a fixed analyst workload, and presents traceable evidence for human review. It keeps AML/fraud and safer-gambling risk indices separate. These 0–100 indices are rankings, not calibrated probabilities, and Sentinel X never freezes an account, restricts play, or contacts a player.

## Quick start

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

macOS/Linux activation: `source .venv/bin/activate`.

Open the local URL printed by Streamlit. The dashboard defaults to 2,000 players, 20,000 responsive-view events, and a review budget equal to 5% of players. The standalone Phase 2 generation command below retains the 100,000-event six-month dataset.

## Verification commands

```powershell
python -m pytest -q
python -m compileall -q .
python demo_cross_channel.py
python validate.py --players 300 --events 6000 --development-seed 142 --evaluation-seed 1042 --budget 50
python data_gen.py --players 2000 --events 100000 --days 182 --seed 42 --output generated_data
python -m ml.train --players 200 --events 4000 --days 182 --seed 42 --model-path artifacts\phase5_models.joblib
```

## Dashboard workspaces

1. **Overview** — live synthetic population, channel, alert, and risk summaries.
2. **Investigation Queue** — capacity-limited ranking with independent AML/fraud and safer-gambling scores.
3. **Player 360** — unified timeline, baselines, peer context, exact score contributions, cited transactions, evidence for/against concern, notes, and audited human dispositions.
4. **Network Intelligence** — typed account relationships and plain-language explanations; shared identifiers alone never establish fraud.
5. **Validation & Fairness** — computed Recall@K, Precision@K, error counts, scenario/channel results, and protected-group audit views.
6. **Red vs Blue Arena** — separate development and held-out synthetic experiments for abstract behavioral adaptation and channel visibility loss.

The sidebar controls synthetic population size, seed, cross-channel visibility, review budget, and displays the active model version. Expensive deterministic preparation and arena runs use Streamlit caches keyed by those settings.

## Data and leakage boundaries

`generate_dataset(...)` returns a `SyntheticGamingData` object. `generate(...)` provides the application adapter `OperationalDataset(profiles, transactions, scenario_labels)`. Scenario labels are stored separately and are rejected by feature, scoring, copilot, and detector boundaries. They are consulted only after scoring for evaluation.

Land-based observations can be unlinked or incomplete. Uncertain identities remain separate source entities; missing records are not converted to zero activity. Age band and region are retained for audit but are not direct scoring inputs.

## Architecture

| Layer | Modules | Responsibility |
|---|---|---|
| Configuration | `configuration.py` | Reproducible seeds and feature/model settings |
| Synthetic environment | `data_gen.py` | Fictional profiles, channel schemas, financial flows, held labels |
| Integration | `cross_channel.py` | Normalization, identity resolution, completeness, deduplication, summaries |
| Features | `feature_engineering.py`, `dashboard_features.py` | Causal histories, robust baselines, peer and cross-channel features |
| Detection | `ml/`, `signals.py`, `models.py` | Isolation Forest, change detection, peer analysis, transparent baselines |
| Network | `network_analysis.py` | Typed weighted graph evidence without risk-score feedback |
| Scoring | `scoring.py` | Independent explainable indices and fixed-capacity prioritization |
| Investigation | `investigator_copilot.py` | Grounded cases, uncertainty, alternatives, local audit trail |
| Assurance | `validate.py`, `red_team.py`, `dashboard_metrics.py` | Held-out metrics, sensitivity, fairness audit, defensive stress experiments |
| Interface | `app.py` | Cached six-workspace Streamlit application |

See [architecture](docs/architecture.md), [model card](docs/model_card.md), [evaluation report](docs/evaluation_report.md), [five-minute demo](docs/demo_script.md), [limitations](docs/limitations.md), and [implementation plan](docs/implementation_plan.md).

## Project structure

```text
sentinel_x/
├── app.py
├── configuration.py
├── data_gen.py
├── cross_channel.py
├── feature_engineering.py
├── dashboard_features.py
├── signals.py
├── models.py
├── scoring.py
├── network_analysis.py
├── investigator_copilot.py
├── dashboard_metrics.py
├── validate.py
├── red_team.py
├── demo_cross_channel.py
├── ml/
├── tests/
├── docs/
└── requirements.txt
```

## Reproducible evaluation snapshot

The detailed report records an actual held-out run with 300 players and 6,000 events in each split, development seed 142, evaluation seed 1042, and a 50-account review budget. At that capacity, cross-channel plus network intelligence measured AML/fraud Recall@50 of 56.25% with 7 false negatives and safer-gambling Recall@50 of 42.86% with 8 false negatives. These are synthetic run results, not expected production performance or proof of superiority.

## Known limitations

- Synthetic labels are injected assumptions, not confirmed fraud or gambling harm.
- Identity, channel availability, and behavioral distributions are simplified.
- The fixed threshold of 60 was poorly matched to several transparent models in the documented run and produced zero threshold alerts; fixed-budget metrics are more useful here.
- Small protected-group positive counts make fairness ratios unstable; equal flag rates would not establish fairness.
- The local session-state case store is not durable, authenticated, or multi-user.
- No production ingestion, monitoring, governance, security controls, or regulatory validation is included.

Future work should validate schemas and labels with authorized domain experts, calibrate thresholds on representative development data, add durable access-controlled case management, test larger temporal holdouts, and conduct formal privacy, security, accessibility, and fairness reviews before any real-world pilot.
