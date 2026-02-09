"""
Executor — safe skill execution with error handling.

Resolves skill names to functions, validates parameters,
executes them, and returns structured results.
"""

import inspect
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..llm.prompts import get_skill_function

logger = logging.getLogger("datapilot.core.executor")


class ExecutionResult:
    """Result of executing a skill."""

    __slots__ = (
        "status", "skill_name", "result", "error",
        "elapsed_seconds", "code_snippet", "columns_used",
    )

    def __init__(
        self,
        status: str,
        skill_name: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        elapsed_seconds: float = 0.0,
        code_snippet: Optional[str] = None,
        columns_used: Optional[list] = None,
    ):
        self.status = status
        self.skill_name = skill_name
        self.result = result
        self.error = error
        self.elapsed_seconds = elapsed_seconds
        self.code_snippet = code_snippet
        self.columns_used = columns_used

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "status": self.status,
            "skill_name": self.skill_name,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        if self.code_snippet is not None:
            d["code_snippet"] = self.code_snippet
        if self.columns_used is not None:
            d["columns_used"] = self.columns_used
        return d


class Executor:
    """Executes analysis skills safely with parameter validation."""

    def execute(
        self,
        skill_name: str,
        df,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a skill function against a DataFrame.

        Args:
            skill_name: Name of the skill (must match a datapilot function).
            df: pandas DataFrame to analyze.
            parameters: Additional keyword arguments for the skill.

        Returns:
            ExecutionResult with status, result dict, or error message.
        """
        parameters = parameters or {}
        start = time.time()

        # Resolve function
        func = get_skill_function(skill_name)
        if func is None:
            return ExecutionResult(
                status="error",
                skill_name=skill_name,
                error=f"Unknown skill: '{skill_name}'",
                elapsed_seconds=time.time() - start,
            )

        # Filter parameters to only those the function accepts
        filtered = self._filter_params(func, df, parameters)

        # Generate code snippet
        code_snippet = self._build_code_snippet(skill_name, filtered)

        # Execute
        try:
            logger.info(f"Executing {skill_name}({list(filtered.keys())})")
            result = func(**filtered)
            elapsed = time.time() - start

            # Validate contract: result must be a dict with "status"
            if not isinstance(result, dict):
                return ExecutionResult(
                    status="error",
                    skill_name=skill_name,
                    error=f"Skill returned {type(result).__name__}, expected dict",
                    elapsed_seconds=elapsed,
                    code_snippet=code_snippet,
                )

            # If result contains a chart_path, read the file and inject base64
            if "chart_path" in result:
                result = self._inject_chart_base64(result)

            # Extract columns used from the result or parameters
            columns_used = self._extract_columns_used(result, filtered, df)

            return ExecutionResult(
                status=result.get("status", "success"),
                skill_name=skill_name,
                result=result,
                elapsed_seconds=elapsed,
                code_snippet=code_snippet,
                columns_used=columns_used,
            )

        except TypeError as e:
            elapsed = time.time() - start
            error_msg = self._humanize_error(skill_name, e, filtered)
            logger.error(f"Skill {skill_name} failed: {e}", exc_info=True)
            return ExecutionResult(
                status="error",
                skill_name=skill_name,
                result={"status": "error", "message": error_msg},
                error=error_msg,
                elapsed_seconds=elapsed,
                code_snippet=code_snippet,
            )
        except KeyError as e:
            elapsed = time.time() - start
            error_msg = f"Missing required column: {e}. Check that the column exists in your dataset."
            logger.error(f"Skill {skill_name} failed: {e}", exc_info=True)
            return ExecutionResult(
                status="error",
                skill_name=skill_name,
                result={"status": "error", "message": error_msg},
                error=error_msg,
                elapsed_seconds=elapsed,
                code_snippet=code_snippet,
            )
        except Exception as e:
            elapsed = time.time() - start
            error_msg = self._humanize_error(skill_name, e, filtered)
            logger.error(f"Skill {skill_name} failed: {e}", exc_info=True)
            return ExecutionResult(
                status="error",
                skill_name=skill_name,
                result={"status": "error", "message": error_msg},
                error=error_msg,
                elapsed_seconds=elapsed,
                code_snippet=code_snippet,
            )

    def _filter_params(
        self,
        func,
        df,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the keyword arguments for a skill function.

        Most skills take `df` or `file_path` as the first positional arg.
        We inspect the signature to pass only valid parameters.
        If the function expects `file_path`, we save the DataFrame to a
        temp CSV and pass the path.
        """
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return {"df": df, **parameters}

        param_names = list(sig.parameters.keys())
        filtered: Dict[str, Any] = {}

        # First parameter is almost always `df`, `data`, or `file_path`
        if param_names:
            first = param_names[0]
            if first in ("df", "data", "dataframe"):
                filtered[first] = df
            elif first in ("file_path", "filepath", "path"):
                # Skill expects a file path — save df to temp CSV
                filtered[first] = self._df_to_temp_path(df)
            elif first in ("text", "image_path"):
                # NLP / OCR skills — don't pass df automatically
                pass
            else:
                # Unknown first param — try passing df
                filtered[first] = df

        # Add matching parameters
        for key, value in parameters.items():
            if key in sig.parameters and key not in filtered:
                filtered[key] = value

        return filtered

    def _df_to_temp_path(self, df) -> str:
        """Save a DataFrame to a temp CSV and return the path."""
        tmp_dir = Path(tempfile.gettempdir()) / "datapilot"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"_exec_{id(df)}.csv"
        df.to_csv(str(tmp_path), index=False)
        return str(tmp_path)

    @staticmethod
    def _inject_chart_base64(result: Dict[str, Any]) -> Dict[str, Any]:
        """Read chart file and inject base64 into result dict."""
        import base64

        chart_path = result.get("chart_path")
        if chart_path:
            path = Path(chart_path)
            if path.exists():
                b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
                result["chart_base64"] = b64
                logger.info(f"Injected chart_base64 from {chart_path}")
        return result

    @staticmethod
    def _humanize_error(
        skill_name: str,
        error: Exception,
        filtered: Dict[str, Any],
    ) -> str:
        """Convert Python exceptions into human-readable error messages."""
        msg = str(error)

        # Missing positional argument
        if "missing" in msg and "required positional argument" in msg:
            # Extract the missing param name
            import re
            match = re.search(r"'(\w+)'", msg)
            param = match.group(1) if match else "unknown"
            if param == "target":
                return (
                    "No target column specified. Please mention which column you want to predict "
                    "(e.g., 'predict churn' or 'classify by status')."
                )
            if param in ("date_column", "date_col"):
                return (
                    "No date column found in the dataset. "
                    "Time series analysis requires a date/datetime column."
                )
            if param in ("value_column", "value_col"):
                return (
                    "No numeric value column specified for forecasting. "
                    "Please mention which numeric column to forecast."
                )
            return f"Missing required parameter: '{param}'. Please include it in your question."

        # Column not found
        if "not found" in msg.lower() or "not in" in msg.lower():
            return msg

        # Generic: clean up the message
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return f"Analysis failed: {msg}"

    @staticmethod
    def _build_code_snippet(
        skill_name: str,
        filtered: Dict[str, Any],
    ) -> str:
        """Generate a Python code snippet showing what was executed."""
        args = []
        for key, val in filtered.items():
            if key in ("df", "data", "dataframe"):
                args.append(f'{key}=df')
            elif key in ("file_path", "filepath", "path"):
                args.append(f'{key}="your_data.csv"')
            elif isinstance(val, str):
                args.append(f'{key}="{val}"')
            elif isinstance(val, (list, tuple)):
                args.append(f'{key}={val!r}')
            else:
                args.append(f'{key}={val!r}')
        args_str = ", ".join(args)
        return f"from datapilot import {skill_name}\nresult = {skill_name}({args_str})"

    @staticmethod
    def _extract_columns_used(
        result: Dict[str, Any],
        filtered: Dict[str, Any],
        df,
    ) -> list:
        """Extract which columns were involved in the analysis."""
        cols = set()

        # From explicit parameters
        for key in ("target", "column", "x", "y", "hue", "date_col", "value_col"):
            val = filtered.get(key)
            if isinstance(val, str):
                cols.add(val)
            elif isinstance(val, (list, tuple)):
                cols.update(str(v) for v in val)

        # From "columns" parameter
        col_param = filtered.get("columns")
        if isinstance(col_param, (list, tuple)):
            cols.update(str(v) for v in col_param)

        # If no explicit columns, skill used all DataFrame columns
        if not cols:
            cols = set(str(c) for c in df.columns)

        # Only return columns that actually exist in the DataFrame
        valid = set(str(c) for c in df.columns)
        return sorted(cols & valid)
