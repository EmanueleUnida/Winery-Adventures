import sys

import numpy as np
import polars as pl
import pytest

from winery_adventures.computations import WineryHPCComputations, pairwise_stress_function
from winery_adventures.main import main, run_full_pipeline
from winery_adventures.pipeline import WineryPipeline
from winery_adventures.transformations import WineryTransformer


def test_transformer_missing_column_raises(sensors_df):
    transformer = WineryTransformer(None)

    with pytest.raises(ValueError, match="pH"):
        transformer.add_avg_ph_per_tank(sensors_df.drop("pH"))
    with pytest.raises(ValueError, match="temp"):
        transformer.add_temperature_deviation(sensors_df.drop("temp"))


def test_transformer_without_tank_info_skips_grape_variety(sensors_df):
    out_df = WineryTransformer(None).analyze_data(sensors_df)

    assert "avg_pH_per_tank" in out_df.columns
    assert "grape_variety_num_readings" not in out_df.columns
    assert out_df.height == sensors_df.height


def test_temperature_deviation_with_zero_quantity_is_null():
    df = pl.DataFrame({"tank_id": [1], "temp": [25.0], "quantity_liters": [0]})

    out_df = WineryTransformer(None).add_temperature_deviation(df)

    assert out_df["temperature_deviation"][0] == 1.0
    assert out_df["temperature_deviation_scaled"][0] is None


def test_stress_of_empty_arrays_is_zero():
    empty = np.array([], dtype=np.float64)

    assert pairwise_stress_function(empty, empty, empty) == 0.0


def test_hpc_missing_column_raises(sensors_df):
    with pytest.raises(ValueError, match="quantity_liters"):
        WineryHPCComputations().analyze_data(sensors_df.drop("quantity_liters"))


def test_hpc_skips_readings_with_invalid_quantity():
    df = pl.DataFrame(
        {
            "tank_id": [1, 1, 1, 1, 2],
            "pH": [3.4, 3.6, 3.9, 3.1, 3.5],
            "temp": [25.0, 26.0, 27.0, 24.0, 26.0],
            "quantity_liters": [500, 500, None, 0, None],
        }
    )

    out_df = WineryHPCComputations(n_jobs=1).analyze_data(df)

    # Tank 1: only the first two readings are valid (same values as the reference test)
    assert out_df["stress_score"][0] == pytest.approx(2.2)
    assert out_df["stress_score"][3] == pytest.approx(2.2)
    # Tank 2: no valid readings, so no stress
    assert out_df["stress_score"][4] == 0.0


def test_pipeline_does_not_log_without_stress_score(monkey_wandb_run):
    pipeline = WineryPipeline([], project_name="TestProj")

    pipeline.log_to_wandb(pl.DataFrame({"tank_id": [1, 2]}))

    assert monkey_wandb_run.logs == []


def test_run_full_pipeline_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_full_pipeline(input_csv=str(tmp_path / "missing.tsv"), output_csv=str(tmp_path / "out.csv"))


def test_run_full_pipeline_without_tank_info(tmp_path, monkey_joblib, monkey_wandb_run, sensors_df):
    sensor_tsv = tmp_path / "sensors.tsv"
    sensor_tsv.write_text(sensors_df.write_csv(separator="\t"))
    output_csv = tmp_path / "results.csv"

    result_df = run_full_pipeline(input_csv=str(sensor_tsv), output_csv=str(output_csv))

    assert result_df.height == sensors_df.height
    assert "grape_variety" not in result_df.columns
    assert "stress_score" in pl.read_csv(output_csv).columns


def test_command_line_writes_results(tmp_path, monkeypatch, monkey_joblib, monkey_wandb_run, sensors_df):
    sensor_tsv = tmp_path / "sensors.tsv"
    sensor_tsv.write_text(sensors_df.write_csv(separator="\t"))
    output_csv = tmp_path / "results.csv"
    monkeypatch.setattr(sys, "argv", ["main", "--input", str(sensor_tsv), "--output", str(output_csv)])

    main()

    assert output_csv.exists()
