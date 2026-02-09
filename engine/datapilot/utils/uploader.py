"""
Upload utility for DataPilot.

Handles uploading results to external storage. Uses DATAPILOT_UPLOAD_PATH
environment variable to locate the upload library. Falls back gracefully
when not available.
"""

import os
from typing import Dict

from .helpers import setup_logging


def upload_result(result: Dict, filename: str) -> None:
    """
    Upload a result dict using the configured upload library.

    Looks for the upload library at the path specified by the
    DATAPILOT_UPLOAD_PATH environment variable. If not set or the
    library is not found, logs a warning and returns silently.
    """
    upload_path = os.environ.get("DATAPILOT_UPLOAD_PATH")
    if not upload_path:
        logger = setup_logging("datapilot.upload")
        logger.warning(
            "DATAPILOT_UPLOAD_PATH not set. Result was not uploaded."
        )
        return

    try:
        import sys
        sys.path.append(upload_path)
        from uploader import upload_data  # type: ignore
        upload_data(result, filename)
    except ImportError:
        logger = setup_logging("datapilot.upload")
        logger.warning(
            "upload_data not available at configured path. "
            "Result was not uploaded."
        )
