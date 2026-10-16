# Five-Minute Hackathon Demo

All screens and numbers use fictional synthetic data. Start with: “Sentinel X is a local investigator decision-support prototype. It does not use or describe OLG internal systems, and it never takes automatic action.”

## 0:00–0:40 — Overview

Run `python -m streamlit run app.py`. Keep the 2,000-player default, seed 42, and cross-channel integration enabled. The review budget is computed as 5% of the population (100 accounts). On **Overview**, point to computed player/event/channel totals, independent high-priority counts, and recent alerts.

## 0:40–1:25 — Cross-channel signal

Run `python demo_cross_channel.py`. The reproducible seed-42 console run generates 240 players and 12,000 normalized events. `PLAYER:P000152` is an injected distributed-cross-channel synthetic case: observed deposits were $26.82 online casino and $34.22 sportsbook, while the unified observed total was $61.04 (1.78× the largest channel). Its legacy priority index changed from 0 in online-casino-only view to 11 in the unified view. This illustrates added context, not proof of wrongdoing or adequate production sensitivity.

## 1:25–2:05 — Prioritization and separate risks

Open **Investigation Queue**. Show that AML/fraud and safer-gambling scores are separate and may diverge. Explain that priority uses the stronger independent review need while the exact budget controls workload.

## 2:05–3:05 — Player 360 and copilot

Open the top queue item in **Player 360**. Show channel history, unified timeline, baseline, peer context, and contribution ledger. Read one item from **EVIDENCE FOR CONCERN** and one from **EVIDENCE AGAINST OR ALTERNATIVE EXPLANATIONS**. Alternatives are unverified hypotheses, missing data is disclosed, and cited transaction IDs can be checked. Demonstrate a note or status control; it records an audit action but takes no adverse action.

## 3:05–4:00 — Red vs Blue Arena

Open **Red vs Blue Arena**, choose both experiments, and run. For the documented 300-player/6,000-event held-out configuration, Blue V2 Recall@50 was 54.17% at Level 0 and 29.17% at Level 3, with 17 false negatives at Level 3. Under uncertain linkage its recall was 41.67%, versus 54.17% under complete visibility. Partial visibility scored 58.33% in this sample; call out that the non-monotonic result is shown honestly and is not a benefit claim.

## 4:00–5:00 — Validation, failures, and close

Open **Validation & Fairness**. At the documented budget of 50, cross-channel plus network AML Recall@50 was 56.25% with 7 false negatives and 41 false positives; safer-gambling Recall@50 was 42.86% with 8 false negatives and 44 false positives. Workload stayed at 50. Call out missed abrupt and distributed cases, then show age/region audit views and the small-sample warning.

Close with: “Sentinel X’s contribution is unified context plus traceable human review. The synthetic evidence shows both detections and failures. Production use would require authorized real-data validation, governance, security, and domain review.”
