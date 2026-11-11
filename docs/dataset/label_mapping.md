# Verified Scenario & Fault Label Mapping

## 1. Scenario-to-Label Mapping Table

The 25 simulated RTU scenarios correspond to verified physical fault injections at defined severity levels, as documented in the LBNL inventory:

| Source Filename | Assigned Fault Class | Severity Level | Binary Label (`is_fault`) | Physical Engineering Description |
|---|---|---|---|---|
| `RTU_sim_baseline.csv` | **FaultFree** | 0.00 | 0 | Baseline normal operation under standard building cooling loads |
| `RTU_sim_condfouling10.csv` | **CondenserFouling** | 0.10 | 1 | Condenser coil airflow reduction by 10% |
| `RTU_sim_condfouling20.csv` | **CondenserFouling** | 0.20 | 1 | Condenser coil airflow reduction by 20% |
| `RTU_sim_condfouling30.csv` | **CondenserFouling** | 0.30 | 1 | Condenser coil airflow reduction by 30% |
| `RTU_sim_condfouling40.csv` | **CondenserFouling** | 0.40 | 1 | Condenser coil airflow reduction by 40% |
| `RTU_sim_condfouling50.csv` | **CondenserFouling** | 0.50 | 1 | Condenser coil airflow reduction by 50% |
| `RTU_sim_evapfouling10.csv` | **EvaporatorFouling** | 0.10 | 1 | Evaporator coil/filter airflow reduction by 10% |
| `RTU_sim_evapfouling20.csv` | **EvaporatorFouling** | 0.20 | 1 | Evaporator coil/filter airflow reduction by 20% |
| `RTU_sim_evapfouling30.csv` | **EvaporatorFouling** | 0.30 | 1 | Evaporator coil/filter airflow reduction by 30% |
| `RTU_sim_evapfouling40.csv` | **EvaporatorFouling** | 0.40 | 1 | Evaporator coil/filter airflow reduction by 40% |
| `RTU_sim_evapfouling50.csv` | **EvaporatorFouling** | 0.50 | 1 | Evaporator coil/filter airflow reduction by 50% |
| `RTU_sim_liquidpipe01bar.csv` | **LiquidLineRestriction** | 0.10 bar | 1 | Liquid line restriction causing 0.1 bar pressure drop |
| `RTU_sim_liquidpipe04bar.csv` | **LiquidLineRestriction** | 0.40 bar | 1 | Liquid line restriction causing 0.4 bar pressure drop |
| `RTU_sim_liquidpipe08bar.csv` | **LiquidLineRestriction** | 0.80 bar | 1 | Liquid line restriction causing 0.8 bar pressure drop |
| `RTU_sim_liquidpipe10bar.csv` | **LiquidLineRestriction** | 1.00 bar | 1 | Liquid line restriction causing 1.0 bar pressure drop |
| `RTU_sim_overcharge10.csv` | **RefrigerantOvercharge** | +10% | 1 | Refrigerant mass charge increased by +10% |
| `RTU_sim_overcharge15.csv` | **RefrigerantOvercharge** | +15% | 1 | Refrigerant mass charge increased by +15% |
| `RTU_sim_overcharge20.csv` | **RefrigerantOvercharge** | +20% | 1 | Refrigerant mass charge increased by +20% |
| `RTU_sim_suctionpipe01bar.csv` | **SuctionLineRestriction** | 0.10 bar | 1 | Suction line pipe restriction causing 0.1 bar pressure drop |
| `RTU_sim_suctionpipe03bar.csv` | **SuctionLineRestriction** | 0.30 bar | 1 | Suction line pipe restriction causing 0.3 bar pressure drop |
| `RTU_sim_suctionpipe06bar.csv` | **SuctionLineRestriction** | 0.60 bar | 1 | Suction line pipe restriction causing 0.6 bar pressure drop |
| `RTU_sim_suctionpipe09bar.csv` | **SuctionLineRestriction** | 0.90 bar | 1 | Suction line pipe restriction causing 0.9 bar pressure drop |
| `RTU_sim_undercharge10.csv` | **RefrigerantUndercharge** | -10% | 1 | Refrigerant mass charge decreased by -10% |
| `RTU_sim_undercharge15.csv` | **RefrigerantUndercharge** | -15% | 1 | Refrigerant mass charge decreased by -15% |
| `RTU_sim_undercharge20.csv` | **RefrigerantUndercharge** | -20% | 1 | Refrigerant mass charge decreased by -20% |

## 2. Supervised Learning Targets

- **Multi-Class Fault Diagnosis:** 7 distinct classes (`FaultFree`, `CondenserFouling`, `EvaporatorFouling`, `LiquidLineRestriction`, `RefrigerantOvercharge`, `SuctionLineRestriction`, `RefrigerantUndercharge`).
- **Binary Fault Detection:** Normal (`is_fault = 0`) vs Faulted (`is_fault = 1`).
- **Strict Rule:** Cluster IDs discovered via unsupervised K-Means or GMM are **never** treated as ground-truth fault diagnosis labels.

