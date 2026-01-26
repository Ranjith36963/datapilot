"""
File loader for ingesting Excel and CSV files into the raw data zone.
"""

import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import RAW_ZONE
from src.utils import setup_logging, MetadataManager
from src.ingestion.schema_inference import SchemaInference


class FileLoader:
    """Loads and ingests files into the raw data zone."""

    SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}

    def __init__(self):
        self.logger = setup_logging("file_loader")
        self.metadata = MetadataManager()
        self.schema_inference = SchemaInference()

    def ingest(
        self,
        source_path: str | Path,
        dataset_name: Optional[str] = None,
        description: str = ""
    ) -> Tuple[Path, pd.DataFrame]:
        """
        Ingest a file into the raw zone.

        Args:
            source_path: Path to the source file
            dataset_name: Optional name for the dataset
            description: Optional description

        Returns:
            Tuple of (destination path, loaded DataFrame)
        """
        source = Path(source_path)

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if source.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {source.suffix}")

        name = dataset_name or source.stem

        # Copy to raw zone (immutable)
        dest_path = RAW_ZONE / source.name
        if dest_path != source:
            shutil.copy2(source, dest_path)
            self.logger.info(f"Copied {source.name} to raw zone")

        # Load data
        df = self._load_file(dest_path)
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Infer and save schema
        schema_path = self.schema_inference.infer_and_save(df, name)

        # Register in metadata
        dataset_id = f"raw_{name.lower().replace(' ', '_')}"
        self.metadata.register_dataset(
            dataset_id=dataset_id,
            name=name,
            zone="raw",
            path=str(dest_path),
            schema_path=str(schema_path),
            row_count=len(df),
            column_count=len(df.columns),
            description=description or f"Raw data file: {source.name}"
        )

        return dest_path, df

    def _load_file(self, path: Path) -> pd.DataFrame:
        """Load a file based on its extension."""
        ext = path.suffix.lower()
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path, engine="openpyxl")
        elif ext == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported extension: {ext}")

    def load_existing(self, filename: str) -> pd.DataFrame:
        """Load an existing file from the raw zone."""
        path = RAW_ZONE / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found in raw zone: {filename}")
        return self._load_file(path)
