# HVAC FaultSense — Data Catalog & Provenance

## 1. Primary Data Source

The primary data source for HVAC FaultSense is the **Lawrence Berkeley National Laboratory (LBNL) Fault Detection and Diagnostics (FDD) Rooftop Unit (RTU) dataset**.

- **Official Portal:** <https://faultdetection.lbl.gov/data/>
- **RTU Equipment Collection:** <https://faultdetection.lbl.gov/dataset/rtu/>
- **Data Catalog (Data.gov):** <https://catalog.data.gov/dataset/lbnl-fault-detection-and-diagnostics-datasets>
- **Scientific Publication:** Granderson et al. (2023), *A labeled dataset for building HVAC systems operating in faulted and fault-free states*, Nature Scientific Data. DOI: [10.1038/s41597-023-02197-w](https://doi.org/10.25984/1881324)

## 2. Directory Hierarchy

```text
data/
├── raw/
│   ├── LBNL_FDD_Data_Sets_RTU.pdf       # Official documentation & equipment report (1.6 MB)
│   ├── LBNL_FDD_Data_Sets_RTU.zip       # Original archive downloaded from LBNL (291.4 MB)
│   ├── metadata/                        # Semantic schemas and ontologies
│   │   ├── LBNL_FDD_Data_Sets_RTU_Sim.ttl # Brick Schema RDF Turtle for Simulated RTU
│   │   └── LBNL_FDD_Data_Sets_RTU_Exp.ttl # Brick Schema RDF Turtle for Experimental RTU
│   ├── simulated/                       # 25 CSV files (143,941 rows each, 1-min sampling)
│   ├── field/                           # 4 CSV files from operational building sites
│   └── laboratory/                      # 52 CSV files from ORNL environmental chamber
├── interim/
│   ├── dataset_manifest.csv             # Full catalog of 81 scenario CSV files and fault severities
│   └── data_dictionary.csv              # 25 sensor attributes with Brick Schema mappings
└── processed/
    ├── test_evaluation_sample.parquet   # Held-out evaluation sample with model predictions
    └── test_evaluation_sample.csv       # CSV fallback for zero-dependency inspection
```

### File Count Reconciliation:
- **Telemetry Scenario CSV Files (Manifest):** $25 \text{ (simulated)} + 4 \text{ (field)} + 52 \text{ (laboratory)} = \mathbf{81} \text{ CSV files}$
- **Semantic Metadata Files:** $2 \text{ Brick Schema Turtle ontologies } (\texttt{.ttl})$
- **Data & Metadata Asset Subtotal:** $81 + 2 = \mathbf{83} \text{ files}$
- **Technical Documentation & Source Archive:** $1 \text{ PDF report} + 1 \text{ ZIP archive} = 2 \text{ files}$
- **Total Raw Files on Disk in `data/raw/`:** $81 + 2 + 1 + 1 = \mathbf{85} \text{ files}$

## 3. Data Integrity & Provenance Rules

1. **No Fabrication:** Telemetry measurements, sensor names, sampling rates (1 minute), and fault scenarios are preserved directly from official files.
2. **Distinguishable Subsets:** Simulated (`data/raw/simulated/`), Field (`data/raw/field/`), and Laboratory (`data/raw/laboratory/`) data are stored separately and never silently blended.
3. **Leakage Prevention:** Normalization, imputation, and feature extraction statistics are computed strictly on the training partitions.

