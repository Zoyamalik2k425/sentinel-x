# Sentinel X Model Card

## Intended use

Sentinel X ranks fictional accounts for analyst review and explains observed evidence. It demonstrates cross-channel investigation concepts for a hackathon. It is not a fraud determination, safer-gambling diagnosis, calibrated probability, production control, or automated decision system.

## Components

- **Isolation Forest:** multivariate anomaly ranking with median imputation and robust scaling.
- **Behavioral change:** rolling median/MAD and robust change signals against the player's prior history.
- **Peer comparison:** historical behavioral group comparison without age or geography as scoring inputs.
- **Network intelligence:** typed shared identifiers, activity overlap, and confirmed ownership. Ownership alone has no investigation strength; shared identifiers are leads, not conclusions.
- **Explainable scoring:** independent 0–100 AML/fraud and safer-gambling indices with configurable normalized weights and exact contribution ledgers.

## Inputs and exclusions

Operational inputs include observed transactions, causal behavioral history, channel coverage, identifier changes, and relationship evidence. Age band and registered region are excluded from direct scoring and retained only for evaluation. Synthetic scenario type, target category, dates, and labels are excluded from training, feature engineering, scoring, and narratives.

## Training and evaluation

Unsupervised models use chronological train/validation/test separation. Scenario labels are used after prediction for threshold and held-out reporting. Phase 11 comparisons use identical held-out populations and fixed review budgets. Development and held-out generator seeds differ. Metrics must be regenerated for each configuration and are not hard-coded into the application.

## Outputs

- AML/fraud review index, 0–100.
- Safer-gambling review index, 0–100.
- Capacity-aware priority rank and risk band.
- Per-signal exact contribution ledger.
- Data completeness and identity-confidence context.
- Supporting synthetic transaction identifiers and deterministic case narrative.

## Risks and safeguards

Sparse or mislinked data can suppress or distort signals. Shared households and payment instruments can create false positives. Novel behavior can produce false negatives. The system discloses missing evidence, keeps uncertain identities separate, shows alternatives as unverified hypotheses, audits group outcomes, and makes no automatic intervention.

## Versioning

The dashboard displays `ModelBundle.model_version`; without a saved artifact it creates a deterministic local fallback using the selected seed. Any real deployment would require governed model/data versioning, access controls, monitoring, and independent validation.
