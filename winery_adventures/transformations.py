"""Column transformations applied to the sensor readings."""

import logging

import polars as pl

from winery_adventures.base import BaseWineryAnalyzer

logger = logging.getLogger(__name__)


class WineryTransformer(BaseWineryAnalyzer):
    """Adds derived columns (averages, counts, temperature deviation).

    Attributes:
        STANDARD_TEMPERATURE: Reference fermentation temperature in Celsius.
        REFERENCE_LITERS: Volume used to scale the temperature deviation.
        tank_info_df: Optional tank information, with ``grape_variety`` already
            split into a list of varieties (one list per tank).
    """

    STANDARD_TEMPERATURE = 26.0
    REFERENCE_LITERS = 1000.0

    def __init__(self, tank_info_df: pl.DataFrame | None = None) -> None:
        """Create the transformer.

        Args:
            tank_info_df: Tank information with columns ``tank_id`` and
                ``grape_variety`` (list of strings). Needed only for the
                per-grape-variety transformation.
        """
        self.tank_info_df = tank_info_df

    @staticmethod
    def _require_columns(df: pl.DataFrame, columns: list[str]) -> None:
        """Raise ``ValueError`` if some of the columns are missing from ``df``."""
        missing = [column for column in columns if column not in df.columns]
        if missing:
            message = f"Missing required column(s): {missing}"
            logger.error(message)
            raise ValueError(message)

    def analyze_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all the transformations in sequence.

        The per-grape-variety step is applied last, and only if tank
        information was provided, because it duplicates each reading once per
        grape variety of its tank.

        Args:
            df: Sensor readings.

        Returns:
            The DataFrame with all the new columns.
        """
        df = self.add_avg_ph_per_tank(df)
        df = self.add_num_readings_per_tank(df)
        df = self.add_temperature_deviation(df)
        if self.tank_info_df is not None:
            df = self.add_num_readings_per_grape_variety(df)
        else:
            logger.info("No tank info provided: skipping grape variety transformation")
        return df

    def add_avg_ph_per_tank(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add ``avg_pH_per_tank``, the mean pH of the tank of each reading.

        Args:
            df: Sensor readings with columns ``tank_id`` and ``pH``.

        Returns:
            The DataFrame with the new column.
        """
        self._require_columns(df, ["tank_id", "pH"])
        return df.with_columns(pl.col("pH").mean().over("tank_id").alias("avg_pH_per_tank"))

    def add_num_readings_per_tank(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add ``tank_num_readings``, the number of readings of each tank.

        Args:
            df: Sensor readings with column ``tank_id``.

        Returns:
            The DataFrame with the new column.
        """
        self._require_columns(df, ["tank_id"])
        return df.with_columns(pl.len().over("tank_id").alias("tank_num_readings"))

    def add_num_readings_per_grape_variety(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add ``grape_variety`` and ``grape_variety_num_readings``.

        Each reading is joined with the grape varieties of its tank, so it
        appears once per variety, and the readings of each variety are counted.

        Args:
            df: Sensor readings with column ``tank_id``.

        Returns:
            The DataFrame with one row per (reading, grape variety).

        Raises:
            AttributeError: If no tank information was provided.
        """
        if self.tank_info_df is None:
            message = "tank_info_df is required to count readings per grape variety"
            logger.error(message)
            raise AttributeError(message)
        self._require_columns(df, ["tank_id"])
        self._require_columns(self.tank_info_df, ["tank_id", "grape_variety"])

        varieties = self.tank_info_df.select("tank_id", "grape_variety")
        return (
            df.join(varieties, on="tank_id", how="left")
            .explode("grape_variety")
            .with_columns(pl.len().over("grape_variety").alias("grape_variety_num_readings"))
        )

    def add_temperature_deviation(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add the deviation of the temperature from ``STANDARD_TEMPERATURE``.

        Always adds ``temperature_deviation`` (absolute deviation). If the
        volume is available it also adds ``temperature_deviation_scaled``, the
        deviation per ``REFERENCE_LITERS`` liters (null when the volume is null
        or not positive).

        Args:
            df: Sensor readings with column ``temp`` and, optionally,
                ``quantity_liters``.

        Returns:
            The DataFrame with the new column(s).
        """
        self._require_columns(df, ["temp"])
        df = df.with_columns((pl.col("temp") - self.STANDARD_TEMPERATURE).abs().alias("temperature_deviation"))
        if "quantity_liters" not in df.columns:
            logger.info("No quantity_liters column: temperature deviation is not scaled")
            return df

        return df.with_columns(
            pl.when(pl.col("quantity_liters") > 0)
            .then(pl.col("temperature_deviation") * self.REFERENCE_LITERS / pl.col("quantity_liters"))
            .alias("temperature_deviation_scaled")
        )
