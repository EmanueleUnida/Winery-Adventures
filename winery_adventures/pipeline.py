"""Pipeline that chains the analyzers and logs the results to wandb."""

import logging
from collections.abc import Sequence

import polars as pl
import wandb

from winery_adventures.base import BaseWineryAnalyzer

logger = logging.getLogger(__name__)


class WineryPipeline:
    """Runs a sequence of analyzers on a DataFrame.

    Attributes:
        analyzers: Objects exposing ``analyze_data(df) -> df``, applied in order.
        project_name: Name of the wandb project used for logging.
    """

    def __init__(self, analyzers: Sequence[BaseWineryAnalyzer], project_name: str = "WineryAdventures") -> None:
        """Create the pipeline.

        Args:
            analyzers: Analyzers to apply, in order.
            project_name: Name of the wandb project.
        """
        self.analyzers = list(analyzers)
        self.project_name = project_name

    def run(self, df: pl.DataFrame, log_to_wandb: bool = False) -> pl.DataFrame:
        """Apply every analyzer in sequence.

        Args:
            df: Input sensor readings.
            log_to_wandb: If True, log the results to wandb at the end.

        Returns:
            The DataFrame produced by the last analyzer.
        """
        for analyzer in self.analyzers:
            logger.info("Running %s", type(analyzer).__name__)
            df = analyzer.analyze_data(df)

        if log_to_wandb:
            self.log_to_wandb(df)
        return df

    def log_to_wandb(self, df: pl.DataFrame) -> None:
        """Log the mean stress score of ``df`` to wandb.

        Nothing is logged if ``df`` has no ``stress_score`` column.

        Args:
            df: DataFrame containing the ``stress_score`` column.
        """
        if "stress_score" not in df.columns:
            logger.warning("No stress_score column: nothing to log to wandb")
            return

        wandb.init(project=self.project_name)
        try:
            wandb.log({"stress_score": df["stress_score"].mean()})
        finally:
            wandb.finish()
