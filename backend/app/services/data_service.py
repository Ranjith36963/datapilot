"""
Data service — file upload handling, temp storage, data loading.
"""

import logging
import os
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger("datapilot.backend.data_service")

# Project-local upload directory (survives OS temp cleanup)
_PROJECT_DATA = Path(__file__).resolve().parents[2] / "data"
UPLOAD_DIR = Path(os.environ.get("DATAPILOT_UPLOAD_DIR", str(_PROJECT_DATA / "uploads")))

# Allowed file extensions
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv"}

# Max file size: 100 MB
MAX_FILE_SIZE = 100 * 1024 * 1024


class DataService:
    """Handles file uploads and temporary storage."""

    def __init__(self, upload_dir: Path | None = None):
        self.upload_dir = upload_dir or UPLOAD_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        filename: str,
        content: bytes,
        session_id: str | None = None,
    ) -> tuple[str, Path]:
        """Save uploaded file to disk.

        Args:
            filename: Original filename.
            content: File bytes.
            session_id: Optional session ID (generated if not provided).

        Returns:
            Tuple of (session_id, file_path).

        Raises:
            ValueError: If file type is not allowed or size exceeds limit.
        """
        # Validate extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # Validate size
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large ({len(content) / 1024 / 1024:.1f} MB). "
                f"Maximum: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
            )

        # Create session directory
        session_id = session_id or uuid.uuid4().hex[:12]
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        safe_name = Path(filename).name  # strip any path components
        file_path = session_dir / safe_name
        file_path.write_bytes(content)

        logger.info(f"Saved upload: {file_path} ({len(content)} bytes)")
        return session_id, file_path

    def get_file_path(self, session_id: str) -> Path | None:
        """Get the uploaded file path for a session."""
        session_dir = self.upload_dir / session_id
        if not session_dir.exists():
            return None
        files = [f for f in session_dir.iterdir() if f.is_file()]
        return files[0] if files else None

    def cleanup_session(self, session_id: str) -> bool:
        """Remove all files for a session."""
        session_dir = self.upload_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info(f"Cleaned up session: {session_id}")
            return True
        return False

    def cleanup_all(self):
        """Remove all uploaded files."""
        if self.upload_dir.exists():
            shutil.rmtree(self.upload_dir)
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleaned up all uploads")

    def load_dataframe(self, file_path: Path):
        """Load a file into a pandas DataFrame."""
        import sys
        # Add engine to path so we can import datapilot
        engine_path = str(Path(__file__).resolve().parents[3] / "engine")
        if engine_path not in sys.path:
            sys.path.insert(0, engine_path)

        from datapilot.utils.helpers import load_data
        return load_data(str(file_path))
