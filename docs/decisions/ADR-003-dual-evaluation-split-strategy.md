# ADR-003: Dual Evaluation Splitting Protocol (Protocols A and B)

## Status
Accepted

## Context
Standard random cross-validation ($k$-fold) randomly mixes consecutive timestamps across train and test partitions. In HVAC thermal systems with high autocorrelation, this introduces severe temporal data leakage where past predictions are informed by temporally adjacent future samples.

Furthermore, an evaluation strategy limited only to in-scenario chronological splitting ensures that models observe every fault scenario and severity level during training, masking whether the diagnostic classifier can generalize out-of-distribution to novel or intermediate fault severities.

## Decision
Implement and report two distinct evaluation protocols:

1. **Protocol A: In-Scenario Chronological Holdout (Temporal Generalization)**
   - For each scenario file, the first 75% of observations chronologically form the training set ($179,925$ records), and the final 25% form the test set ($60,000$ records).
   - Guarantees zero look-ahead bias and evaluates model ability to forecast future time steps under known equipment operating regimes.
   - Scalers are fitted exclusively on the chronological training split and applied to test data.

2. **Protocol B: Unseen Scenario Severity Holdout (Out-of-Distribution Generalization)**
   - Completely withholds 4 entire scenario datasets from training:
     - `RTU_sim_condfouling30.csv` (30% condenser coil fouling)
     - `RTU_sim_evapfouling30.csv` (30% evaporator coil fouling)
     - `RTU_sim_undercharge15.csv` (15% refrigerant undercharge)
     - `RTU_sim_overcharge15.csv` (15% refrigerant overcharge)
   - Evaluates whether models trained on base fault severities (e.g. 10%, 20%, 40%) can correctly classify intermediate, previously unobserved fault severities ($201,537$ train, $38,388$ test).

## Consequences
- **Positive:**
  - Evaluates both temporal forecast stability and diagnostic generalization to uncalibrated fault degrees.
  - CART demonstrated 71.86% accuracy / 0.5830 Macro-F1 on Protocol A, and 57.77% accuracy / 0.4954 Macro-F1 on Protocol B, demonstrating solid robustness.
- **Negative / Trade-offs:**
  - Requires maintaining two separate test evaluation partitions and recording both results in benchmark reporting.

