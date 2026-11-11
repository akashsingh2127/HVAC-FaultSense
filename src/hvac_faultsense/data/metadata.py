"""Metadata catalog, dataset manifest, and data dictionary management.

Maps official LBNL RTU files to verified scenarios, fault types, severity,
and provides physical engineering definitions from the Brick Schema ontology.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Verified mapping for Simulated RTU filenames to official fault categories and severities
SIMULATED_SCENARIO_MAPPING = {
    "RTU_sim_baseline.csv": {
        "fault_type": "FaultFree",
        "severity": 0.0,
        "is_fault": 0,
        "description": "Baseline fault-free operation under normal building loads",
    },
    "RTU_sim_condfouling10.csv": {
        "fault_type": "CondenserFouling",
        "severity": 0.10,
        "is_fault": 1,
        "description": "Condenser coil fouling with 10% airflow reduction",
    },
    "RTU_sim_condfouling20.csv": {
        "fault_type": "CondenserFouling",
        "severity": 0.20,
        "is_fault": 1,
        "description": "Condenser coil fouling with 20% airflow reduction",
    },
    "RTU_sim_condfouling30.csv": {
        "fault_type": "CondenserFouling",
        "severity": 0.30,
        "is_fault": 1,
        "description": "Condenser coil fouling with 30% airflow reduction",
    },
    "RTU_sim_condfouling40.csv": {
        "fault_type": "CondenserFouling",
        "severity": 0.40,
        "is_fault": 1,
        "description": "Condenser coil fouling with 40% airflow reduction",
    },
    "RTU_sim_condfouling50.csv": {
        "fault_type": "CondenserFouling",
        "severity": 0.50,
        "is_fault": 1,
        "description": "Condenser coil fouling with 50% airflow reduction",
    },
    "RTU_sim_evapfouling10.csv": {
        "fault_type": "EvaporatorFouling",
        "severity": 0.10,
        "is_fault": 1,
        "description": "Evaporator coil / filter fouling with 10% airflow reduction",
    },
    "RTU_sim_evapfouling20.csv": {
        "fault_type": "EvaporatorFouling",
        "severity": 0.20,
        "is_fault": 1,
        "description": "Evaporator coil / filter fouling with 20% airflow reduction",
    },
    "RTU_sim_evapfouling30.csv": {
        "fault_type": "EvaporatorFouling",
        "severity": 0.30,
        "is_fault": 1,
        "description": "Evaporator coil / filter fouling with 30% airflow reduction",
    },
    "RTU_sim_evapfouling40.csv": {
        "fault_type": "EvaporatorFouling",
        "severity": 0.40,
        "is_fault": 1,
        "description": "Evaporator coil / filter fouling with 40% airflow reduction",
    },
    "RTU_sim_evapfouling50.csv": {
        "fault_type": "EvaporatorFouling",
        "severity": 0.50,
        "is_fault": 1,
        "description": "Evaporator coil / filter fouling with 50% airflow reduction",
    },
    "RTU_sim_liquidpipe01bar.csv": {
        "fault_type": "LiquidLineRestriction",
        "severity": 0.10,
        "is_fault": 1,
        "description": "Liquid line pipe restriction causing 0.1 bar pressure drop",
    },
    "RTU_sim_liquidpipe04bar.csv": {
        "fault_type": "LiquidLineRestriction",
        "severity": 0.40,
        "is_fault": 1,
        "description": "Liquid line pipe restriction causing 0.4 bar pressure drop",
    },
    "RTU_sim_liquidpipe08bar.csv": {
        "fault_type": "LiquidLineRestriction",
        "severity": 0.80,
        "is_fault": 1,
        "description": "Liquid line pipe restriction causing 0.8 bar pressure drop",
    },
    "RTU_sim_liquidpipe10bar.csv": {
        "fault_type": "LiquidLineRestriction",
        "severity": 1.00,
        "is_fault": 1,
        "description": "Liquid line pipe restriction causing 1.0 bar pressure drop",
    },
    "RTU_sim_overcharge10.csv": {
        "fault_type": "RefrigerantOvercharge",
        "severity": 0.10,
        "is_fault": 1,
        "description": "Refrigerant overcharge by +10% above nominal charge",
    },
    "RTU_sim_overcharge15.csv": {
        "fault_type": "RefrigerantOvercharge",
        "severity": 0.15,
        "is_fault": 1,
        "description": "Refrigerant overcharge by +15% above nominal charge",
    },
    "RTU_sim_overcharge20.csv": {
        "fault_type": "RefrigerantOvercharge",
        "severity": 0.20,
        "is_fault": 1,
        "description": "Refrigerant overcharge by +20% above nominal charge",
    },
    "RTU_sim_suctionpipe01bar.csv": {
        "fault_type": "SuctionLineRestriction",
        "severity": 0.10,
        "is_fault": 1,
        "description": "Suction line pipe restriction causing 0.1 bar pressure drop",
    },
    "RTU_sim_suctionpipe03bar.csv": {
        "fault_type": "SuctionLineRestriction",
        "severity": 0.30,
        "is_fault": 1,
        "description": "Suction line pipe restriction causing 0.3 bar pressure drop",
    },
    "RTU_sim_suctionpipe06bar.csv": {
        "fault_type": "SuctionLineRestriction",
        "severity": 0.60,
        "is_fault": 1,
        "description": "Suction line pipe restriction causing 0.6 bar pressure drop",
    },
    "RTU_sim_suctionpipe09bar.csv": {
        "fault_type": "SuctionLineRestriction",
        "severity": 0.90,
        "is_fault": 1,
        "description": "Suction line pipe restriction causing 0.9 bar pressure drop",
    },
    "RTU_sim_undercharge10.csv": {
        "fault_type": "RefrigerantUndercharge",
        "severity": 0.10,
        "is_fault": 1,
        "description": "Refrigerant undercharge by -10% below nominal charge",
    },
    "RTU_sim_undercharge15.csv": {
        "fault_type": "RefrigerantUndercharge",
        "severity": 0.15,
        "is_fault": 1,
        "description": "Refrigerant undercharge by -15% below nominal charge",
    },
    "RTU_sim_undercharge20.csv": {
        "fault_type": "RefrigerantUndercharge",
        "severity": 0.20,
        "is_fault": 1,
        "description": "Refrigerant undercharge by -20% below nominal charge",
    },
}

# Verified sensor dictionary referencing Brick Schema RDF (LBNL_FDD_Data_Sets_RTU_Sim.ttl)
SIMULATED_SENSOR_DICTIONARY = [
    {
        "column_name": "Datetime",
        "physical_interpretation": "Timestamp of minute-level sampling measurement",
        "measurement_unit": "ISO 8601 YYYY-MM-DD HH:MM:SS",
        "data_type": "datetime64[ns]",
        "brick_class": "brick:Timestamp",
        "subsystem": "Temporal",
        "role": "Index / Time Coordinate",
    },
    {
        "column_name": "RTU_COMP_WATT",
        "physical_interpretation": "Electric power consumed by the refrigeration compressor",
        "measurement_unit": "Watts (W)",
        "data_type": "float64",
        "brick_class": "brick:Electrical_Power_Sensor",
        "subsystem": "Compressor",
        "role": "Subsystem Power Feature",
    },
    {
        "column_name": "RTU_OA_FLOW",
        "physical_interpretation": "Volumetric airflow rate of incoming outdoor air",
        "measurement_unit": "cfm / m^3/s",
        "data_type": "float64",
        "brick_class": "brick:Outside_Air_Flow_Sensor",
        "subsystem": "Economizer / Ventilation",
        "role": "Operating Feature",
    },
    {
        "column_name": "RTU_OA_HUM",
        "physical_interpretation": "Outdoor ambient air relative humidity",
        "measurement_unit": "% RH",
        "data_type": "float64",
        "brick_class": "brick:Outside_Air_Humidity_Sensor",
        "subsystem": "Ambient Environment",
        "role": "Weather Boundary Feature",
    },
    {
        "column_name": "RTU_OA_TEMP",
        "physical_interpretation": "Outdoor ambient drybulb air temperature",
        "measurement_unit": "deg F / deg C",
        "data_type": "float64",
        "brick_class": "brick:Outside_Air_Temperature_Sensor",
        "subsystem": "Ambient Environment",
        "role": "Weather Boundary Feature",
    },
    {
        "column_name": "RTU_RA_FLOW",
        "physical_interpretation": "Volumetric airflow rate of return air from the building zone",
        "measurement_unit": "cfm / m^3/s",
        "data_type": "float64",
        "brick_class": "brick:Return_Air_Flow_Sensor",
        "subsystem": "Air Distribution",
        "role": "Operating Feature",
    },
    {
        "column_name": "RTU_RA_HUM",
        "physical_interpretation": "Relative humidity of return air entering the RTU",
        "measurement_unit": "% RH",
        "data_type": "float64",
        "brick_class": "brick:Return_Air_Humidity_Sensor",
        "subsystem": "Air Distribution",
        "role": "Operating Feature",
    },
    {
        "column_name": "RTU_RA_TEMP",
        "physical_interpretation": "Temperature of return air entering the RTU",
        "measurement_unit": "deg F / deg C",
        "data_type": "float64",
        "brick_class": "brick:Return_Air_Temperature_Sensor",
        "subsystem": "Air Distribution",
        "role": "Operating Feature",
    },
    {
        "column_name": "RTU_REFG_COND_PRES",
        "physical_interpretation": "Refrigerant pressure at the condenser coil",
        "measurement_unit": "psig / kPa",
        "data_type": "float64",
        "brick_class": "brick:Pressure_Sensor",
        "subsystem": "Condenser",
        "role": "Refrigerant Cycle Feature",
    },
    {
        "column_name": "RTU_REFG_COND_TEMP",
        "physical_interpretation": "Refrigerant saturation/condensing temperature at condenser",
        "measurement_unit": "deg F / deg C",
        "data_type": "float64",
        "brick_class": "brick:Temperature_Sensor",
        "subsystem": "Condenser",
        "role": "Refrigerant Cycle Feature",
    },
    {
        "column_name": "RTU_REFG_DISC_PRES",
        "physical_interpretation": "Refrigerant high-side pressure at compressor discharge",
        "measurement_unit": "psig / kPa",
        "data_type": "float64",
        "brick_class": "brick:Pressure_Sensor",
        "subsystem": "Compressor",
        "role": "Refrigerant Cycle Feature",
    },
    {
        "column_name": "RTU_REFG_DISC_TEMP",
        "physical_interpretation": "Refrigerant vapor temperature at compressor discharge line",
        "measurement_unit": "deg F / deg C",
        "data_type": "float64",
        "brick_class": "brick:Discharge_Air_Temperature_Sensor",
        "subsystem": "Compressor",
        "role": "Refrigerant Cycle Feature",
    },
    {
        "column_name": "RTU_REFG_SUCT_PRES",
        "physical_interpretation": "Refrigerant low-side pressure at compressor suction line",
        "measurement_unit": "psig / kPa",
        "data_type": "float64",
        "brick_class": "brick:Pressure_Sensor",
        "subsystem": "Compressor",
        "role": "Refrigerant Cycle Feature",
    },
    {
        "column_name": "RTU_REFG_SUCT_TEMP",
        "physical_interpretation": "Refrigerant suction gas temperature entering compressor",
        "measurement_unit": "deg F / deg C",
        "data_type": "float64",
        "brick_class": "brick:Temperature_Sensor",
        "subsystem": "Compressor",
        "role": "Refrigerant Cycle Feature",
    },
    {
        "column_name": "RTU_SA_FAN_WATT",
        "physical_interpretation": "Electric power consumed by the supply air blower fan",
        "measurement_unit": "Watts (W)",
        "data_type": "float64",
        "brick_class": "brick:Electrical_Power_Sensor",
        "subsystem": "Supply Fan",
        "role": "Subsystem Power Feature",
    },
    {
        "column_name": "RTU_SA_FLOW",
        "physical_interpretation": "Total volumetric supply airflow delivered to zone",
        "measurement_unit": "cfm / m^3/s",
        "data_type": "float64",
        "brick_class": "brick:Supply_Air_Flow_Sensor",
        "subsystem": "Air Distribution",
        "role": "Operating Feature",
    },
    {
        "column_name": "RTU_SA_HUM",
        "physical_interpretation": "Relative humidity of treated supply air leaving RTU",
        "measurement_unit": "% RH",
        "data_type": "float64",
        "brick_class": "brick:Supply_Air_Humidity_Sensor",
        "subsystem": "Air Distribution",
        "role": "Operating Feature",
    },
    {
        "column_name": "RTU_SA_TEMP",
        "physical_interpretation": "Temperature of treated supply air leaving cooling coil",
        "measurement_unit": "deg F / deg C",
        "data_type": "float64",
        "brick_class": "brick:Supply_Air_Temperature_Sensor",
        "subsystem": "Air Distribution",
        "role": "Operating Feature",
    },
    {
        "column_name": "RTU_SEN_CAPA",
        "physical_interpretation": "Sensible cooling capacity provided by RTU",
        "measurement_unit": "Btu/h or Watts",
        "data_type": "float64",
        "brick_class": "brick:coolingCapacity",
        "subsystem": "Thermodynamics",
        "role": "Capacity Metric",
    },
    {
        "column_name": "RTU_STG_STA",
        "physical_interpretation": "Compressor staging operational state (0=Off, 1=Stage 1, 2=Stage 2)",
        "measurement_unit": "Discrete State {0, 1, 2}",
        "data_type": "float64 / int",
        "brick_class": "brick:On_Off_Status",
        "subsystem": "Control",
        "role": "Operating Mode Feature",
    },
    {
        "column_name": "RTU_TOT_CAPA",
        "physical_interpretation": "Total (sensible + latent) cooling capacity delivered by RTU",
        "measurement_unit": "Btu/h or Watts",
        "data_type": "float64",
        "brick_class": "brick:coolingCapacity",
        "subsystem": "Thermodynamics",
        "role": "Capacity Metric",
    },
    {
        "column_name": "RTU_TOT_WATT",
        "physical_interpretation": "Total electric power demand of the entire rooftop unit",
        "measurement_unit": "Watts (W)",
        "data_type": "float64",
        "brick_class": "brick:Electrical_Power_Sensor",
        "subsystem": "System Electrical",
        "role": "Continuous Regression Target",
    },
    {
        "column_name": "ZA_HUM",
        "physical_interpretation": "Zone indoor air relative humidity",
        "measurement_unit": "% RH",
        "data_type": "float64",
        "brick_class": "brick:Zone_Air_Humidity_Sensor",
        "subsystem": "Thermal Zone",
        "role": "Zone Condition Feature",
    },
    {
        "column_name": "ZA_TEMP",
        "physical_interpretation": "Zone indoor drybulb temperature",
        "measurement_unit": "deg F / deg C",
        "data_type": "float64",
        "brick_class": "brick:Zone_Air_Temperature_Sensor",
        "subsystem": "Thermal Zone",
        "role": "Zone Condition Feature",
    },
    {
        "column_name": "ZA_TEMP_SPT",
        "physical_interpretation": "Zone indoor temperature setpoint specified by thermostat",
        "measurement_unit": "deg F / deg C",
        "data_type": "float64",
        "brick_class": "brick:Zone_Air_Temperature_Setpoint",
        "subsystem": "Thermal Zone",
        "role": "Thermostat Target Feature",
    },
]


def build_dataset_manifest(
    raw_sim_dir: Path,
    raw_field_dir: Optional[Path] = None,
    raw_lab_dir: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Scan raw directories and build a verified dataset manifest."""
    records = []

    # 1. Simulated RTU files
    if raw_sim_dir.exists():
        for csv_file in sorted(raw_sim_dir.glob("*.csv")):
            fname = csv_file.name
            meta = SIMULATED_SCENARIO_MAPPING.get(
                fname,
                {
                    "fault_type": "Unknown",
                    "severity": 0.0,
                    "is_fault": -1,
                    "description": "Unclassified scenario",
                },
            )
            file_size_kb = round(csv_file.stat().st_size / 1024, 2)
            records.append({
                "source_file": fname,
                "relative_path": str(csv_file.relative_to(raw_sim_dir.parent.parent)),
                "dataset_group": "Simulated",
                "equipment_type": "Rooftop Unit (RTU)",
                "scenario_name": fname.replace(".csv", ""),
                "fault_type": meta["fault_type"],
                "fault_severity": meta["severity"],
                "is_fault": meta["is_fault"],
                "description": meta["description"],
                "file_size_kb": file_size_kb,
            })

    # 2. Field RTU files
    if raw_field_dir and raw_field_dir.exists():
        for csv_file in sorted(raw_field_dir.glob("*.csv")):
            fname = csv_file.name
            is_fault = 0 if "unfaulted" in fname.lower() else 1
            fault_type = "FaultFree" if is_fault == 0 else "FieldFault"
            if "staging" in fname.lower():
                fault_type = "StagingFault"
            elif "undercharged" in fname.lower():
                fault_type = "RefrigerantUndercharge"

            records.append({
                "source_file": fname,
                "relative_path": str(csv_file.relative_to(raw_field_dir.parent.parent)),
                "dataset_group": "Field",
                "equipment_type": "Rooftop Unit (RTU)",
                "scenario_name": fname.replace(".csv", ""),
                "fault_type": fault_type,
                "fault_severity": -1.0,
                "is_fault": is_fault,
                "description": f"Field operational test: {fname}",
                "file_size_kb": round(csv_file.stat().st_size / 1024, 2),
            })

    # 3. Laboratory (ORNL) RTU files
    if raw_lab_dir and raw_lab_dir.exists():
        for csv_file in sorted(raw_lab_dir.glob("*.csv")):
            fname = csv_file.name
            is_fault = 0 if "unfaulted" in fname.lower() or fname.startswith("ERTU_") else 1
            fault_type = "Baseline" if is_fault == 0 else "LabFault"
            records.append({
                "source_file": fname,
                "relative_path": str(csv_file.relative_to(raw_lab_dir.parent.parent)),
                "dataset_group": "Laboratory",
                "equipment_type": "Flexible Research RTU (FRP)",
                "scenario_name": fname.replace(".csv", ""),
                "fault_type": fault_type,
                "fault_severity": -1.0,
                "is_fault": is_fault,
                "description": f"ORNL environmental chamber test: {fname}",
                "file_size_kb": round(csv_file.stat().st_size / 1024, 2),
            })

    df_manifest = pd.DataFrame(records)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_manifest.to_csv(output_csv, index=False)
        print(f"Dataset manifest written to {output_csv} ({len(df_manifest)} entries).")

    return df_manifest


def get_data_dictionary(output_csv: Optional[Path] = None) -> pd.DataFrame:
    """Return the official sensor data dictionary dataframe."""
    df_dict = pd.DataFrame(SIMULATED_SENSOR_DICTIONARY)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_dict.to_csv(output_csv, index=False)
        print(f"Data dictionary written to {output_csv} ({len(df_dict)} entries).")
    return df_dict


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[3]
    manifest = build_dataset_manifest(
        raw_sim_dir=base_dir / "data" / "raw" / "simulated",
        raw_field_dir=base_dir / "data" / "raw" / "field",
        raw_lab_dir=base_dir / "data" / "raw" / "laboratory",
        output_csv=base_dir / "data" / "interim" / "dataset_manifest.csv",
    )
    data_dict = get_data_dictionary(output_csv=base_dir / "data" / "interim" / "data_dictionary.csv")

