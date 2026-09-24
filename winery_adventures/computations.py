"""Numba-optimised computations for the Winery Adventures pipeline."""

import logging

import joblib
import numpy as np
import polars as pl
from numba import njit

from winery_adventures.base import BaseWineryAnalyzer

logger = logging.getLogger(__name__)


@njit(nogil=True)
def pairwise_stress_function(pH_vals: np.ndarray, temp_vals: np.ndarray, quantity_vals: np.ndarray) -> float:
    """Compute the fermentation stress over all pairs of readings, O(n^2).

    For every pair of readings (i, j) the pH deviation and the temperature
    deviation (weighted by 2) are combined and multiplied by a factor that is
    larger for small tanks, which are thermally less stable. The result is
    the average over the n * n pairs.

    The function is compiled with Numba (``nogil=True`` lets several tanks be
    processed in parallel threads).

    Args:
        pH_vals: pH of each reading.
        temp_vals: Temperature of each reading.
        quantity_vals: Volume of must in liters of each reading (must be > 0).

    Returns:
        The overall stress, or 0.0 if there are no readings.
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


class WineryHPCComputations(BaseWineryAnalyzer):
    """Adds the fermentation stress score of each tank to the readings.

    The score is computed independently for every tank, in parallel with
    Joblib, and repeated on every reading of that tank.

    Attributes:
        n_jobs: Number of parallel workers (-1 uses all the cores).
    """

    REQUIRED_COLUMNS = ("tank_id", "pH", "temp", "quantity_liters")

    def __init__(self, n_jobs: int = -1) -> None:
        """Create the analyzer.

        Args:
            n_jobs: Number of parallel workers used by Joblib.
        """
        self.n_jobs = n_jobs

    @staticmethod
    def _compute_tank_stress(tank_id: int, tank_df: pl.DataFrame, results: dict) -> None:
        """Compute the stress of one tank and store it in ``results``.

        Readings with a missing or non-positive volume cannot be used in the
        formula and are skipped (a warning is logged). A tank without valid
        readings gets a score of 0.0.

        Args:
            tank_id: Identifier of the tank.
            tank_df: Readings of the tank.
            results: Dictionary where the score is stored, keyed by tank id.
        """
        valid_df = tank_df.drop_nulls(["pH", "temp", "quantity_liters"]).filter(pl.col("quantity_liters") > 0)
        skipped = tank_df.height - valid_df.height
        if skipped:
            logger.warning("Tank %s: skipped %d reading(s) with missing or invalid values", tank_id, skipped)

        results[tank_id] = pairwise_stress_function(
            valid_df["pH"].to_numpy(),
            valid_df["temp"].to_numpy(),
            valid_df["quantity_liters"].to_numpy(),
        )

    def analyze_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add the ``stress_score`` column.

        Args:
            df: Sensor readings with columns ``tank_id``, ``pH``, ``temp`` and
                ``quantity_liters``.

        Returns:
            The DataFrame with the new ``stress_score`` column.

        Raises:
            ValueError: If some required columns are missing.
        """
        missing = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            message = f"Missing required column(s): {missing}"
            logger.error(message)
            raise ValueError(message)

        # Workers write into a shared dict: threads (sharedmem) are enough
        # because the Numba function releases the GIL.
        results: dict = {}
        joblib.Parallel(n_jobs=self.n_jobs, require="sharedmem")(
            joblib.delayed(self._compute_tank_stress)(tank_id, tank_df, results)
            for (tank_id,), tank_df in df.group_by("tank_id")
        )

        scores = pl.DataFrame(
            {"tank_id": list(results.keys()), "stress_score": list(results.values())},
            schema={"tank_id": df.schema["tank_id"], "stress_score": pl.Float64},
        )
        return df.join(scores, on="tank_id", how="left")
