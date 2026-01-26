"""
Utility functions for the Veritly data platform.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import REGISTRY_PATH, LINEAGE_PATH


def setup_logging(name: str = "veritly") -> logging.Logger:
    """Set up logging with console output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def generate_version() -> str:
    """Generate a version string based on timestamp."""
    return datetime.now().strftime("v%Y%m%d_%H%M%S")


def get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.now().isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file, return empty dict if not exists."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


class MetadataManager:
    """Manages dataset registry and lineage tracking."""

    def __init__(self):
        self.logger = setup_logging("metadata")
        self.registry = load_json(REGISTRY_PATH)
        self.lineage = load_json(LINEAGE_PATH)

        if "datasets" not in self.registry:
            self.registry["datasets"] = {}
        if "flows" not in self.lineage:
            self.lineage["flows"] = []

    def register_dataset(
        self,
        dataset_id: str,
        name: str,
        zone: str,
        path: str,
        schema_path: Optional[str] = None,
        row_count: int = 0,
        column_count: int = 0,
        description: str = ""
    ) -> None:
        """Register a dataset in the registry."""
        self.registry["datasets"][dataset_id] = {
            "name": name,
            "zone": zone,
            "path": str(path),
            "schema_path": str(schema_path) if schema_path else None,
            "row_count": row_count,
            "column_count": column_count,
            "description": description,
            "created_at": get_timestamp(),
            "version": generate_version()
        }
        self._save_registry()
        self.logger.info(f"Registered dataset: {dataset_id} in {zone} zone")

    def add_lineage(
        self,
        source_id: str,
        target_id: str,
        transformation: str,
        details: Optional[Dict] = None
    ) -> None:
        """Track data lineage between datasets."""
        flow = {
            "source": source_id,
            "target": target_id,
            "transformation": transformation,
            "details": details or {},
            "timestamp": get_timestamp()
        }
        self.lineage["flows"].append(flow)
        self._save_lineage()
        self.logger.info(f"Added lineage: {source_id} -> {target_id} ({transformation})")

    def _save_registry(self) -> None:
        save_json(self.registry, REGISTRY_PATH)

    def _save_lineage(self) -> None:
        save_json(self.lineage, LINEAGE_PATH)
