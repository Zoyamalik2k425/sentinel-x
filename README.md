# Sentinel X — ThunderHacks 2026

Sentinel X is a local, explainable prototype for cross-channel player-protection and fraud-investigation prioritization. It uses only fictional synthetic data and makes no claim about OLG's data, systems, controls, or architecture.

The foundation addresses four demonstration needs: unusual-activity signals, contextual assessment against player history, fixed-capacity investigation prioritization, and an evidence-based investigator view. Recall and workload are evaluated together at a configurable top-K review budget; no model-performance result is hard-coded or claimed in advance.

## Quick start

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS or Linux, create the environment with `python3.12 -m venv .venv`, activate it with `source .venv/bin/activate`, and run the same `pip` and `streamlit` module commands.

Run the tests with:

```powershell
python -m pytest -q
```

Run the Phase 3 console comparison with:

```powershell
python demo_cross_channel.py
```

The demonstration compares the largest individual-channel deposit view with the unified observed total. It does not launch or modify the dashboard.

## Generate the synthetic environment

The default generation is deterministic and creates 2,000 fictional adult players, 100,000 events over 182 days, and scenario assignments for 9% of players:

```powershell
python data_gen.py --players 2000 --events 100000 --days 182 --seed 42 --output generated_data
```

The command writes gzip-compressed CSV files for player profiles, online-casino events, sports-betting events, land-based events, and held scenario labels, plus `metadata.json` containing validation results. These are synthetic project schemas and are not representations of OLG internal databases.

Programmatic callers can use `generate_dataset(...)`, `validate_dataset(...)`, and `save_dataset(...)`. The `generate(...)` application adapter returns an `OperationalDataset` with named `profiles`, `transactions`, and `scenario_labels` fields. Labels are never merged into operational events.

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Configuration | `configuration.py` | Fixed seeds, feature windows, model weights, and review cutoffs |
| Synthetic data | `data_gen.py` | Fictional profiles, transactions, and a separate ground-truth table |
| Cross-channel integration | `cross_channel.py` | Common events, conservative identity resolution, completeness, deduplication, and unified summaries |
| Feature engineering | `signals.py` | Historical/recent, loss, geographic, channel, coverage, and network signals |
| Models | `models.py` | Separate transparent AML/fraud and safer-gambling model outputs |
| Risk scoring | `scoring.py` | Review indices, workload priority, evidence, and suggested human action |
| Evaluation | `validate.py`, `red_team.py` | Fixed-budget metrics, group audit, and missing-observation stress tests |
| Investigator UI | `app.py` | Queue, Player 360, stress arena, and offline validation views |

Synthetic `scenario` and `injected_risk` fields exist only in the ground-truth table. Operational feature engineering rejects a profile frame containing those columns. The investigator queue and Player 360 view do not reveal them.

## Current scope

Implemented:

- Reproducible, three-channel fictional histories with explicit observation gaps.
- Contextual signals comparing recent behavior with observed history.
- Confirmed, probable, unlinked, and conflicting identity states; only confirmed matches receive canonical IDs.
- Channel-specific and unified activity summaries that retain missingness and observation quality.
- Separate AML/fraud and safer-gambling outputs; scores are indices, not probabilities.
- A priority queue that uses the higher of the two review indices to support high recall.
- Evidence and uncertainty statements, data-quality indicators, and human-only dispositions.
- Computed precision/recall at a fixed investigation budget, stress tests, and descriptive group flag rates.

Deliberately not implemented in this foundation phase:

- Production ingestion, identity resolution, authentication, persistence, or investigator workflow integration.
- A fitted anomaly detector, calibrated classifier, graph-learning model, causal fairness analysis, or model registry.
- Automated adverse action, real-time decisions, or claims of compliance effectiveness.

## Limitations

- Ground truth is injected simulation metadata, not confirmed financial crime or gambling harm.
- The network model is a simple shared-token signal; NetworkX is reserved for the next graph-analysis phase.
- Sparse observations can suppress signals. Missing activity is never treated as evidence of safety.
- Group flag-rate differences are descriptive and do not by themselves establish bias or fairness.
- Results depend on the generator assumptions and seed and do not generalize to real players.

See [the implementation plan](docs/implementation_plan.md) for the audit and phased roadmap.
