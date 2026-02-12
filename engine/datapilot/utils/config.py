"""
Configuration for DataPilot.

Provides centralized configuration with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """DataPilot configuration."""

    # Paths
    UPLOAD_PATH: Optional[str] = os.environ.get("DATAPILOT_UPLOAD_PATH")
    TEMP_DIR: str = os.environ.get("DATAPILOT_TEMP_DIR", "/tmp/datapilot")

    # LLM
    LLM_PROVIDER: str = os.environ.get("DATAPILOT_LLM_PROVIDER", "groq")
    OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ANTHROPIC_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
    GROQ_API_KEY: Optional[str] = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    # Analysis defaults
    DEFAULT_CV_FOLDS: int = 5
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_RANDOM_STATE: int = 42
    MAX_SAMPLE_SIZE: int = 50000

    # Export
    BRAND_NAME: str = os.environ.get("DATAPILOT_BRAND_NAME", "DataPilot")
    REPORT_TITLE: str = os.environ.get("DATAPILOT_REPORT_TITLE", "Data Analysis Report")

    @classmethod
    def ensure_temp_dir(cls) -> Path:
        """Create and return the temp directory."""
        p = Path(cls.TEMP_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p
