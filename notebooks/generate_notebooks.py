"""Generates official clean Jupyter notebooks for exploratory profiling and experiment analysis."""

import json
from pathlib import Path


def create_profiling_notebook(dest_path: Path):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# HVAC FaultSense: Notebook 01 - Dataset Profiling and Quality Audit\n",
                    "\n",
                    "This notebook loads the official LBNL Rooftop Unit (RTU) dataset, generates the Brick Schema data dictionary, computes descriptive statistics, and visualizes physical sensor distributions.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from pathlib import Path\n",
                    "from hvac_faultsense.data.ingestion import load_simulated_dataset\n",
                    "from hvac_faultsense.data.metadata import build_dataset_manifest, get_data_dictionary\n",
                    "from hvac_faultsense.data.validation import profile_dataset, generate_profile_markdown\n",
                    "from hvac_faultsense.visualization.plots import plot_missing_values, plot_correlation_heatmap\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1. Load Data Dictionary and Dataset Manifest"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "base_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "manifest = build_dataset_manifest(base_dir / 'data' / 'raw' / 'simulated')\n",
                    "data_dict = get_data_dictionary()\n",
                    "print(f'Cataloged {len(manifest)} scenarios.')\n",
                    "data_dict.head(10)",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Ingest and Statistically Profile Telemetry"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = load_simulated_dataset(base_dir / 'data' / 'raw' / 'simulated', stride=30)\n",
                    "print('Loaded shape:', df.shape)\n",
                    "profile = profile_dataset(df)\n",
                    "pd.DataFrame(profile['statistics']).T.head(15)",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3. Correlation Heatmap"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "corr_df = pd.DataFrame(profile['correlations'])\n",
                    "fig_corr = plot_correlation_heatmap(corr_df)\n",
                    "fig_corr.show()",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(dest_path, "w") as f:
        json.dump(nb, f, indent=2)


def create_experiment_notebook(dest_path: Path):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# HVAC FaultSense: Notebook 02 - Experiment Analysis and Model Benchmarking\n",
                    "\n",
                    "Covers from-scratch regression (OLS & Ridge), unsupervised regime discovery (PCA & K-Means), sequential state tracking (HMM-Viterbi), and fault classification (CART & AdaBoost).",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from pathlib import Path\n",
                    "from hvac_faultsense.pipeline import run_pipeline\n",
                    "import plotly.express as px\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1. Load Benchmark Summary Table"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "base_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "bench_path = base_dir / 'reports' / 'results' / 'benchmark_summary.csv'\n",
                    "df_bench = pd.read_csv(bench_path)\n",
                    "df_bench",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Inspect Sample Evaluation Predictions"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sample_path = base_dir / 'data' / 'processed' / 'test_evaluation_sample.csv'\n",
                    "df_sample = pd.read_csv(sample_path)\n",
                    "df_sample[['Datetime', 'RTU_TOT_WATT', 'pred_power_ols', 'residual_ols', 'fault_type', 'pred_fault_cart']].head(10)",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(dest_path, "w") as f:
        json.dump(nb, f, indent=2)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    create_profiling_notebook(base / "notebooks" / "01_dataset_profiling.ipynb")
    create_experiment_notebook(base / "notebooks" / "02_experiment_analysis.ipynb")
    print("Notebooks generated successfully.")

