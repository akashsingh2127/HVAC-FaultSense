# LBNL RTU Dataset Overview & Provenance

## 1. Official Dataset Details

- **Dataset Title:** LBNL Fault Detection and Diagnostics Datasets: Rooftop Units (RTUs)
- **Institution:** Lawrence Berkeley National Laboratory (LBNL), Oak Ridge National Laboratory (ORNL), and National Renewable Energy Laboratory (NREL).
- **Official Access Portal:** <https://faultdetection.lbl.gov/dataset/rtu/>
- **Equipment Type:** Commercial Rooftop Air Conditioning Unit (RTU) with dual compressor stages, economizer outdoor air damper, supply blower fan, evaporator cooling coil, and air-cooled condenser.
- **Sampling Frequency:** Exactly 1 minute (60 seconds) continuous time-series interval.
- **License / Terms of Use:** Public open-access research dataset sponsored by the U.S. Department of Energy (DOE).

## 2. Dataset Groups

The LBNL RTU resource consists of three distinct experimental groupings:

1. **Simulated RTU Collection (`data/raw/simulated/`):**
   - 25 CSV scenario files (291.4 MB).
   - Generated using validated physics-based simulation models representing realistic seasonal building thermal loads.
   - Exactly 143,941 rows per scenario, spanning from `2018-07-20 01:00:00` to `2018-10-28 00:00:00`.
   - Total records: 3,598,525 observations across 25 identical physical sensor points.
   - Zero missing values.

2. **Field RTU Collection (`data/raw/field/`):**
   - 4 CSV scenario files (`Site1_Staging_Fault.csv`, `Site1_Unfaulted.csv`, `Site2_Undercharged40.csv`, `Site2_Unfaulted.csv`).
   - Recorded from physical commercial buildings operating under real occupancy and weather.

3. **Laboratory RTU Collection (`data/raw/laboratory/`):**
   - 52 CSV files collected in the Oak Ridge National Laboratory (ORNL) Flexible Research Platform environmental chambers across Fall 2020, Spring 2021, Summer 2021, and Winter 2022.
   - Contains 57 measured thermodynamic and control channels.

## 3. Discovered Schema vs Preliminary Assumptions

Earlier project descriptions hypothesized arbitrary columns such as `high_pressure_1`, `low_pressure_1`, and `co2_1`.
Inspection of the real LBNL files confirms that the actual 25 columns follow standard ASHRAE and Brick Schema conventions (`RTU_REFG_DISC_PRES`, `RTU_REFG_SUCT_PRES`, `RTU_OA_TEMP`, `RTU_TOT_WATT`). No sensor points or column names have been fabricated.

