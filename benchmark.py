"""Performance benchmark of the fermentation stress computation (time and memory).

Usage:
    python data_generator.py    # creates data/full_sensors.tsv and data/full_tank_info.tsv
    python benchmark.py         # runs the benchmark, logs to wandb, writes docs/performance_report.md

Set ``WANDB_MODE=offline`` to run without a wandb account.
"""

import argparse
import logging
import math
import os
import platform
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numba
import numpy as np
import polars as pl
import psutil
import wandb

from winery_adventures.computations import WineryHPCComputations, pairwise_stress_function
from winery_adventures.transformations import WineryTransformer

SCALING_SIZES = [250, 500, 1000, 2000, 4000]
DEFAULT_REPORT = Path("docs") / "performance_report.md"


def stress_pure_python(pH_vals: list, temp_vals: list, quantity_vals: list) -> float:
    """Fermentation stress in plain Python: same algorithm as the Numba version (baseline).

    Args:
        pH_vals: pH of each reading.
        temp_vals: Temperature of each reading.
        quantity_vals: Volume in liters of each reading.

    Returns:
        The overall stress.
    """
    n = len(pH_vals)
    if n == 0:
        return 0.0
    stress_sum = 0.0
    for i in range(n):
        for j in range(n):
            pH_dev = abs(pH_vals[i] - pH_vals[j])
            t_dev = abs(temp_vals[i] - temp_vals[j]) * 2.0
            quantity_factor = (500.0 / quantity_vals[i]) + (500.0 / quantity_vals[j])
            stress_sum += (pH_dev + t_dev) * quantity_factor
    return stress_sum / (n * n)


def best_time(func: Callable, *args, repeats: int = 1) -> float:
    """Run ``func(*args)`` several times and return the shortest time in seconds.

    Args:
        func: Function to time.
        *args: Arguments of the function.
        repeats: Number of runs.

    Returns:
        The best (shortest) run time in seconds.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return min(times)


class PeakMemory:
    """Context manager that measures the peak resident memory (RSS) of the process in a block.

    The memory is sampled by a background thread, so that memory allocated by
    native libraries (Polars, Numba) is measured too, unlike with ``tracemalloc``.
    """

    def __init__(self, interval: float = 0.005) -> None:
        """Create the meter.

        Args:
            interval: Seconds between two samples.
        """
        self._process = psutil.Process(os.getpid())
        self._interval = interval
        self._stop = threading.Event()
        self.baseline = 0
        self.peak = 0

    def _sample(self) -> None:
        """Keep the highest RSS seen until the block ends."""
        while not self._stop.is_set():
            self.peak = max(self.peak, self._process.memory_info().rss)
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakMemory":
        """Start sampling."""
        self.baseline = self.peak = self._process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        """Stop sampling."""
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._process.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        """Peak RSS of the process in MB."""
        return self.peak / 1024**2

    @property
    def increase_mb(self) -> float:
        """Peak RSS minus the RSS at the start of the block, in MB."""
        return (self.peak - self.baseline) / 1024**2


def measure(name: str, func: Callable) -> tuple[object, dict]:
    """Run ``func()`` measuring time and memory.

    Args:
        name: Name of the stage.
        func: Function without arguments to run.

    Returns:
        The result of ``func`` and a dictionary with the measurements.
    """
    with PeakMemory() as memory:
        start = time.perf_counter()
        result = func()
        seconds = time.perf_counter() - start
    return result, {
        "stage": name,
        "seconds": seconds,
        "peak_mb": memory.peak_mb,
        "increase_mb": memory.increase_mb,
    }


def run_scaling(sensors_df: pl.DataFrame) -> tuple[float, list[dict]]:
    """Compare plain Python and Numba on increasing numbers of readings.

    Args:
        sensors_df: Sensor readings.

    Returns:
        The Numba compilation time in seconds and one dictionary of results per size.
    """
    valid = sensors_df.drop_nulls(["pH", "temp", "quantity_liters"]).filter(pl.col("quantity_liters") > 0)
    pH, temp, quantity = (valid[column].to_numpy() for column in ("pH", "temp", "quantity_liters"))

    start = time.perf_counter()
    pairwise_stress_function(pH[:10], temp[:10], quantity[:10])  # first call compiles the function
    compile_seconds = time.perf_counter() - start

    rows = []
    for n in SCALING_SIZES:
        if n > len(pH):
            break
        arrays = pH[:n], temp[:n], quantity[:n]
        numba_seconds = best_time(pairwise_stress_function, *arrays, repeats=5)
        lists = [array.tolist() for array in arrays]
        python_seconds = best_time(stress_pure_python, *lists)
        assert math.isclose(pairwise_stress_function(*arrays), stress_pure_python(*lists), rel_tol=1e-9)
        rows.append(
            {
                "n": n,
                "python_s": python_seconds,
                "numba_s": numba_seconds,
                "speedup": python_seconds / numba_seconds,
            }
        )
        print(f"n={n}: python {python_seconds:.3f}s, numba {numba_seconds:.5f}s")
    return compile_seconds, rows


def run_pipeline(sensors_tsv: str, tank_info_tsv: str) -> list[dict]:
    """Measure the stages of the pipeline on the full dataset.

    Args:
        sensors_tsv: Path of the sensors file.
        tank_info_tsv: Path of the tank info file.

    Returns:
        One dictionary of measurements per stage.
    """

    def load() -> tuple[pl.DataFrame, pl.DataFrame]:
        sensors = pl.read_csv(sensors_tsv, separator="\t")
        tank_info = pl.read_csv(tank_info_tsv, separator="\t").with_columns(pl.col("grape_variety").str.split(","))
        return sensors, tank_info

    (sensors_df, tank_info_df), loading = measure("Load TSV files", load)
    transformed_df, transformation = measure(
        "Transformations", lambda: WineryTransformer(tank_info_df).analyze_data(sensors_df)
    )
    single_df, single = measure(
        "Stress, n_jobs=1", lambda: WineryHPCComputations(n_jobs=1).analyze_data(transformed_df)
    )
    parallel_df, parallel = measure(
        f"Stress, n_jobs=-1 ({os.cpu_count()} cores)",
        lambda: WineryHPCComputations(n_jobs=-1).analyze_data(transformed_df),
    )
    assert single_df["stress_score"].equals(parallel_df["stress_score"]), "Parallel result differs from serial"

    stages = [loading, transformation, single, parallel]
    for stage in stages:
        stage["rows"] = transformed_df.height if stage["stage"] != "Load TSV files" else sensors_df.height
        print(f"{stage['stage']}: {stage['seconds']:.3f}s, peak {stage['peak_mb']:.0f} MB")
    return stages


def machine_info() -> dict:
    """Describe the machine and the library versions used for the measurements.

    Returns:
        Dictionary with the description.
    """
    return {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": platform.python_version(),
        "numba": numba.__version__,
        "polars": pl.__version__,
        "numpy": np.__version__,
    }


def log_to_wandb(project: str, info: dict, compile_seconds: float, scaling: list[dict], stages: list[dict]) -> None:
    """Log all the measurements to wandb.

    Args:
        project: Name of the wandb project.
        info: Machine description.
        compile_seconds: Numba compilation time.
        scaling: Results of the scaling comparison.
        stages: Results of the pipeline stages.
    """
    wandb.init(project=project, config=info)
    try:
        wandb.log({"numba_compile_seconds": compile_seconds})
        for row in scaling:
            wandb.log(
                {
                    "n_readings": row["n"],
                    "python_seconds": row["python_s"],
                    "numba_seconds": row["numba_s"],
                    "speedup": row["speedup"],
                }
            )
        for stage in stages:
            key = stage["stage"]
            wandb.log(
                {
                    f"{key}/seconds": stage["seconds"],
                    f"{key}/peak_mb": stage["peak_mb"],
                    f"{key}/increase_mb": stage["increase_mb"],
                }
            )
    finally:
        wandb.finish()


def write_report(path: Path, info: dict, compile_seconds: float, scaling: list[dict], stages: list[dict]) -> None:
    """Write the performance report in Markdown, with the measured numbers.

    Args:
        path: Output file.
        info: Machine description.
        compile_seconds: Numba compilation time.
        scaling: Results of the scaling comparison.
        stages: Results of the pipeline stages.
    """
    last = scaling[-1]
    single, parallel = stages[2], stages[3]
    parallel_ratio = single["seconds"] / parallel["seconds"]
    growth = scaling[-1]["numba_s"] / scaling[-2]["numba_s"]
    total_seconds = stages[1]["seconds"] + min(single["seconds"], parallel["seconds"])

    lines = [
        "# Performance report",
        "",
        "Generated by `benchmark.py`. All numbers are measured on the machine described below.",
        "",
        "## Setup",
        "",
        f"- OS: {info['os']}",
        f"- CPU: {info['cpu']} ({info['cores']} logical cores), RAM: {info['ram_gb']} GB",
        f"- Python {info['python']}, Numba {info['numba']}, Polars {info['polars']}, NumPy {info['numpy']}",
        f"- Dataset: {stages[0]['rows']:,} sensor readings; "
        f"{stages[1]['rows']:,} rows after the grape variety transformation",
        "",
        "## 1. Stress formula: plain Python vs Numba",
        "",
        "The formula compares every pair of readings, so its cost is O(n^2). "
        "Best time over several runs, without compilation time.",
        "",
        "| Readings (n) | Plain Python (s) | Numba (s) | Speedup |",
        "|---:|---:|---:|---:|",
    ]
    lines += [f"| {r['n']:,} | {r['python_s']:.3f} | {r['numba_s']:.5f} | {r['speedup']:,.0f}x |" for r in scaling]
    lines += [
        "",
        f"- Numba compiles the function at the first call: {compile_seconds:.2f} s, paid once per process.",
        f"- With n = {last['n']:,} readings Numba is {last['speedup']:,.0f} times faster than plain Python.",
        f"- Doubling n from {scaling[-2]['n']:,} to {last['n']:,} multiplies the Numba time by {growth:.1f} "
        "(about 4 is expected for an O(n^2) algorithm).",
        "",
        "## 2. Full pipeline on the whole dataset",
        "",
        "| Stage | Time (s) | Peak memory (MB) | Memory increase (MB) |",
        "|---|---:|---:|---:|",
    ]
    lines += [f"| {s['stage']} | {s['seconds']:.3f} | {s['peak_mb']:.0f} | {s['increase_mb']:.0f} |" for s in stages]
    verdict = "faster" if parallel_ratio >= 1 else "slower"
    lines += [
        "",
        "Peak memory is the resident memory (RSS) of the whole process, sampled while the stage runs; "
        "it includes the data loaded by the previous stages.",
        "",
        f"- Transformations plus stress score take about {total_seconds:.2f} s for the whole dataset.",
        f"- The stress is computed independently for each tank. With Joblib the run with n_jobs=-1 is "
        f"{max(parallel_ratio, 1 / parallel_ratio):.2f}x {verdict} than with n_jobs=1.",
        '- Joblib uses threads (`require="sharedmem"`): the Numba function is compiled with `nogil=True`, '
        "so the threads really run in parallel and no data is copied between processes.",
        "",
        "## Conclusions",
        "",
        "Compiling the O(n^2) stress formula with Numba is what makes the analysis practical: "
        f"the same algorithm is about {last['speedup']:,.0f} times faster than plain Python. "
        + (
            f"Parallelising over tanks with Joblib gives a further gain ({parallel_ratio:.2f}x), "
            "limited by the number of cores and by the tanks with more readings."
            if parallel_ratio >= 1.1
            else "Parallelising over tanks with Joblib gives no clear gain on this machine."
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the benchmark, log it to wandb and write the report."""
    parser = argparse.ArgumentParser(description="Benchmark of the fermentation stress computation")
    parser.add_argument("--sensors", default="data/full_sensors.tsv", help="sensors TSV file")
    parser.add_argument("--tank-info", default="data/full_tank_info.tsv", help="tank info TSV file")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="output Markdown report")
    parser.add_argument("--project", default="WineryAdventures-Benchmark", help="wandb project name")
    args = parser.parse_args()

    # The per-tank warnings about readings without volume would flood the output.
    logging.getLogger("winery_adventures").setLevel(logging.ERROR)

    for path in (args.sensors, args.tank_info):
        if not Path(path).is_file():
            raise SystemExit(f"File not found: {path}. Run 'python data_generator.py' first.")

    info = machine_info()
    sensors_df = pl.read_csv(args.sensors, separator="\t")
    compile_seconds, scaling = run_scaling(sensors_df)
    stages = run_pipeline(args.sensors, args.tank_info)

    log_to_wandb(args.project, info, compile_seconds, scaling, stages)
    write_report(Path(args.report), info, compile_seconds, scaling, stages)
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
