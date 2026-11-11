# ADR-001: LBNL RTU Dataset Selection and Temporal Sampling Strategy

## Status
Accepted

## Context
The HVAC FaultSense project requires an authoritative, physically grounded, and publicly verifiable dataset for fault detection, diagnostics, and baseline energy modeling of packaged rooftop units (RTUs). 

Synthetic toy datasets or fabricated sensor columns fail to capture non-linear thermodynamic interactions (e.g., suction superheat, condenser subcooling, refrigerant pressure-enthalpy shifts). Conversely, raw 1-minute telemetry across 25 simulated scenarios totals 3,598,525 observations, requiring significant memory footprint and extended CPU computation time during pairwise clustering and iterative boosting.

## Decision
1. **Source:** We selected the official **Lawrence Berkeley National Laboratory (LBNL) RTU Fault Detection and Diagnostics Dataset** (DOI: 10.25984/1881324), specifically the simulated RTU dataset spanning 25 calibrated scenarios.
2. **Schema Grounding:** All 25 physical sensor columns and control variables are strictly cataloged against the official Brick Schema RDF ontology (`LBNL_FDD_Data_Sets_RTU_Sim.ttl`) and official technical documentation. No columns or labels are fabricated.
3. **Temporal Sampling Strategy:** We adopted a fixed 15-minute uniform stride (`stride=15`), reducing the dataset from 3,598,525 to 239,925 observations (9,597 records per scenario).

## Consequences
- **Positive:**
  - Preserves macroscopic diurnal weather shifts, zone load variations, and steady-state thermodynamic operating regimes. Typical building thermal time constants are 30 to 120 minutes, well within the Nyquist rate of 15-minute sampling.
  - Ensures all 5 ML units (including NumPy from-scratch CART and AdaBoost) can train, validate, and test in under 2 minutes with deterministic execution.
  - Leaves zero missing values and zero duplicate records across the entire processed corpus.
- **Negative / Trade-offs:**
  - Sub-minute transient phenomena (e.g., instantaneous compressor startup inrush current, fast thermal expansion valve hunting cycles) are smoothed out.
  - This downsampling is explicitly documented in the dataset overview and configuration files.

