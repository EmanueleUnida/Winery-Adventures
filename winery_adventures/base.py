"""Abstract base class shared by every Winery Adventures analyzer."""

from abc import ABC, abstractmethod

import polars as pl


class BaseWineryAnalyzer(ABC):
    """Common interface for all the steps of the winery data pipeline.

    Each concrete analyzer receives a Polars DataFrame with the sensor
    readings and returns a (possibly transformed) DataFrame.
    """

    @abstractmethod
    def analyze_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Analyze or transform the given data.

        Args:
            df: DataFrame with the sensor readings.

        Returns:
            The transformed DataFrame.
        """
