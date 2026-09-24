"""Entry point: load the data, run the pipeline and save the results."""

import argparse
import logging
from pathlib import Path

import polars as pl

from winery_adventures.computations import WineryHPCComputations
from winery_adventures.pipeline import WineryPipeline
from winery_adventures.transformations import WineryTransformer

logger = logging.getLogger(__name__)


def _read_tsv(path: str) -> pl.DataFrame:
    """Read a tab-separated file.

    Args:
        path: Path of the file.

    Returns:
        The file content.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not Path(path).is_file():
        message = f"File not found: {path}"
        logger.error(message)
        raise FileNotFoundError(message)
    return pl.read_csv(path, separator="\t")


def run_full_pipeline(
    input_csv: str,
    tank_info_csv: str | None = None,
    output_csv: str = "results.csv",
    project_name: str = "WineryAdventures",
) -> pl.DataFrame:
    """Load the TSV files, run transformations and HPC analysis, save the result.

    Args:
        input_csv: Path of the sensors TSV file.
        tank_info_csv: Optional path of the tank info TSV file.
        output_csv: Path where the resulting CSV is written.
        project_name: Name of the wandb project used for logging.

    Returns:
        The final DataFrame.
    """
    sensors_df = _read_tsv(input_csv)

    tank_info_df = None
    if tank_info_csv is not None:
        tank_info_df = _read_tsv(tank_info_csv).with_columns(pl.col("grape_variety").str.split(","))

    pipeline = WineryPipeline([WineryTransformer(tank_info_df), WineryHPCComputations()], project_name=project_name)
    result_df = pipeline.run(sensors_df, log_to_wandb=True)

    result_df.write_csv(output_csv)
    logger.info("Saved %d rows to %s", result_df.height, output_csv)
    return result_df


def main() -> None:
    """Command line interface of the pipeline."""
    parser = argparse.ArgumentParser(description="Winery Adventures data pipeline")
    parser.add_argument("--input", required=True, help="sensors TSV file")
    parser.add_argument("--tank-info", default=None, help="tank info TSV file (optional)")
    parser.add_argument("--output", default="results.csv", help="output CSV file")
    parser.add_argument("--project", default="WineryAdventures", help="wandb project name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_full_pipeline(args.input, args.tank_info, args.output, args.project)


if __name__ == "__main__":
    main()
