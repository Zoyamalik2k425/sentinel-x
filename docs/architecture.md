# Sentinel X Architecture

## Scope and principles

Sentinel X is a local investigator decision-support prototype over fictional synthetic data. This architecture is a project design, not a description of OLG systems. Labels never enter operational detection, uncertain identities are not silently merged, missing observations are not treated as zero activity, and AML/fraud remains separate from safer gambling.

## Data flow

```text
Synthetic channel generators
  ├─ online casino
  ├─ online sportsbook
  └─ land casino (partly linked / partly observed)
          │
          ▼
Normalization + conservative identity resolution + deduplication
          │
          ├─ completeness and channel summaries
          ├─ causal feature history ──► anomaly / change / peer models
          └─ typed identifier evidence ──► relationship graph
                                      │
                                      ▼
                         independent explainable risk indices
                                      │
                                      ▼
                   fixed-capacity queue + grounded investigation case
                                      │
                                      ▼
                     Streamlit decision-support workspaces

Separate held scenario labels ──► post-score validation only
```

## Key contracts

- `SyntheticGamingData` retains distinct channel tables and a separate `scenario_labels` table.
- `OperationalDataset` is the stable three-part adapter: `profiles`, normalized `transactions`, and held `scenario_labels`.
- `CrossChannelResult` carries normalized events, identity evidence, completeness, summaries, and deduplication counts.
- Feature frames are event-time causal: an event at time T uses information available at or before T.
- `ModelBundle` persists fitted anomaly and peer components with training cutoffs and version metadata.
- Scoring returns separate indices, contribution ledgers, data coverage, identity confidence, and supporting event IDs.
- `InvestigationCase` converts supplied operational evidence into deterministic prose and human workflow state.
- `ValidationResult` contains held-out overall, scenario, channel, sensitivity, and fairness tables.

## Identity and missingness

Confirmed identities map to canonical players. Probable, conflicting, and unlinked records preserve their source entity and confidence; they do not automatically enter confirmed history. Completeness fields record observed channels, missing fields, known unobserved records, and observation gaps. Aggregations use unknown/null where a value cannot be calculated reliably.

## Performance and caching

The UI builds one deterministic dashboard resource per `(player_count, seed, cross_channel)` setting. Model preparation, integration, feature histories, network layout, and scores reuse that cached resource. Arena results use a separate cache keyed by population, seed, and review budget. Network visualization is capped while graph statistics retain the full graph.

## Trust boundaries

Protected attributes are audit-only. Final risk scores and alert outcomes never feed back into network features. Labels are rejected at operational boundaries. Recommendations require a human decision and prohibit automatic adverse action.
