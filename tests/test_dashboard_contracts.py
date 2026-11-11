"""Test suite verifying contracts, serialized artifacts, and data schemas expected by the Streamlit frontend."""

import json
from pathlib import Path
import pandas as pd
import pytest


def test_metadata_artifacts_exist():
    base_dir = Path(__file__).resolve().parents[1]
    manifest_path = base_dir / "data" / "interim" / "dataset_manifest.csv"
    dict_path = base_dir / "data" / "interim" / "data_dictionary.csv"

    assert manifest_path.exists(), "Dataset manifest must exist."
    assert dict_path.exists(), "Data dictionary must exist."

    df_manifest = pd.read_csv(manifest_path)
    df_dict = pd.read_csv(dict_path)

    assert len(df_manifest) >= 25
    assert len(df_dict) == 25
    assert "fault_type" in df_manifest.columns
    assert "brick_class" in df_dict.columns


def test_reports_and_benchmark_results():
    base_dir = Path(__file__).resolve().parents[1]
    bench_path = base_dir / "reports" / "results" / "benchmark_summary.csv"
    results_path = base_dir / "reports" / "results" / "experiment_results.json"

    assert bench_path.exists()
    assert results_path.exists()

    df_bench = pd.read_csv(bench_path)
    assert not df_bench.empty
    expected_tasks = {"Regression", "Clustering", "Classification"}
    assert expected_tasks.issubset(set(df_bench["Task"].unique()))

    with open(results_path, "r") as f:
        res = json.load(f)
    assert "best_models" in res
    assert "classification_metrics" in res
    assert "regression_metrics" in res

