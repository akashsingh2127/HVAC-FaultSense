# HVAC FaultSense — Verified Sensor Data Dictionary

Every telemetry channel is mapped directly to its physical equipment subsystem, engineering unit, and Brick Schema ontology class (`LBNL_FDD_Data_Sets_RTU_Sim.ttl`).

| Column Name | Physical Interpretation | Measurement Unit | Data Type | Brick Ontology Class | Subsystem | Analytic Role |
|---|---|---|---|---|---|---|
| `Datetime` | Sampling timestamp | ISO 8601 (YYYY-MM-DD HH:MM:SS) | `datetime64[ns]` | `brick:Timestamp` | Temporal | Time Coordinate |
| `RTU_COMP_WATT` | Compressor electric power demand | Watts (W) | `float64` | `brick:Electrical_Power_Sensor` | Compressor | Power Feature |
| `RTU_OA_FLOW` | Outdoor air ventilation flow rate | cfm | `float64` | `brick:Outside_Air_Flow_Sensor` | Economizer | Operating Feature |
| `RTU_OA_HUM` | Outdoor ambient relative humidity | % RH | `float64` | `brick:Outside_Air_Humidity_Sensor` | Ambient | Weather Boundary |
| `RTU_OA_TEMP` | Outdoor ambient drybulb air temperature | °F / °C | `float64` | `brick:Outside_Air_Temperature_Sensor` | Ambient | Weather Boundary |
| `RTU_RA_FLOW` | Return air airflow from conditioned zone | cfm | `float64` | `brick:Return_Air_Flow_Sensor` | Air Loop | Operating Feature |
| `RTU_RA_HUM` | Return air relative humidity | % RH | `float64` | `brick:Return_Air_Humidity_Sensor` | Air Loop | Operating Feature |
| `RTU_RA_TEMP` | Return air drybulb temperature | °F / °C | `float64` | `brick:Return_Air_Temperature_Sensor` | Air Loop | Operating Feature |
| `RTU_REFG_COND_PRES` | Refrigerant condensing pressure | psig / kPa | `float64` | `brick:Pressure_Sensor` | Condenser | Refrigerant Cycle |
| `RTU_REFG_COND_TEMP` | Refrigerant condensing saturation temperature | °F / °C | `float64` | `brick:Temperature_Sensor` | Condenser | Refrigerant Cycle |
| `RTU_REFG_DISC_PRES` | Refrigerant compressor discharge high-side pressure | psig / kPa | `float64` | `brick:Pressure_Sensor` | Compressor | Refrigerant Cycle |
| `RTU_REFG_DISC_TEMP` | Refrigerant compressor discharge gas temperature | °F / °C | `float64` | `brick:Discharge_Air_Temperature_Sensor` | Compressor | Refrigerant Cycle |
| `RTU_REFG_SUCT_PRES` | Refrigerant compressor suction low-side pressure | psig / kPa | `float64` | `brick:Pressure_Sensor` | Compressor | Refrigerant Cycle |
| `RTU_REFG_SUCT_TEMP` | Refrigerant compressor suction gas temperature | °F / °C | `float64` | `brick:Temperature_Sensor` | Compressor | Refrigerant Cycle |
| `RTU_SA_FAN_WATT` | Supply air blower fan electric power | Watts (W) | `float64` | `brick:Electrical_Power_Sensor` | Supply Fan | Power Feature |
| `RTU_SA_FLOW` | Supply air volumetric airflow rate | cfm | `float64` | `brick:Supply_Air_Flow_Sensor` | Air Loop | Operating Feature |
| `RTU_SA_HUM` | Supply air relative humidity leaving cooling coil | % RH | `float64` | `brick:Supply_Air_Humidity_Sensor` | Air Loop | Operating Feature |
| `RTU_SA_TEMP` | Supply air drybulb temperature leaving cooling coil | °F / °C | `float64` | `brick:Supply_Air_Temperature_Sensor` | Air Loop | Operating Feature |
| `RTU_SEN_CAPA` | Sensible cooling capacity delivered to building | Watts / Btu/h | `float64` | `brick:coolingCapacity` | Thermodynamics | Capacity Metric |
| `RTU_STG_STA` | Compressor staging operational status | Discrete {0, 1, 2} | `float64` | `brick:On_Off_Status` | Control | Operating State |
| `RTU_TOT_CAPA` | Total cooling capacity (sensible + latent) | Watts / Btu/h | `float64` | `brick:coolingCapacity` | Thermodynamics | Capacity Metric |
| `RTU_TOT_WATT` | Total electric power demand of entire RTU | Watts (W) | `float64` | `brick:Electrical_Power_Sensor` | Electrical | **Regression Target** |
| `ZA_HUM` | Zone indoor air relative humidity | % RH | `float64` | `brick:Zone_Air_Humidity_Sensor` | Thermal Zone | Thermal Zone |
| `ZA_TEMP` | Zone indoor drybulb air temperature | °F / °C | `float64` | `brick:Zone_Air_Temperature_Sensor` | Thermal Zone | Thermal Zone |
| `ZA_TEMP_SPT` | Zone indoor thermostat temperature setpoint | °F / °C | `float64` | `brick:Zone_Air_Temperature_Setpoint` | Thermal Zone | Control Target |

