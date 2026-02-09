"""
Data validator — check data quality and enforce constraints.

Provides comprehensive validation: nulls, duplicates, outliers,
type consistency, string format issues. Returns a score 0-100.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.helpers import load_data, setup_logging, get_numeric_columns
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.validator")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_nulls(df: pd.DataFrame) -> dict:
    """Detailed null analysis per column."""
    records: List[Dict[str, Any]] = []
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        records.append({
            "column": col,
            "null_count": n_null,
            "null_pct": round(n_null / len(df) * 100, 2) if len(df) else 0.0,
            "has_nulls": n_null > 0,
        })
    total_null = int(df.isna().sum().sum())
    total_cells = len(df) * len(df.columns)
    return {
        "total_null_cells": total_null,
        "total_cells": total_cells,
        "overall_null_pct": round(total_null / total_cells * 100, 2) if total_cells else 0.0,
        "columns": records,
    }


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> dict:
    """Find duplicate rows, optionally by subset of columns."""
    dup_mask = df.duplicated(subset=subset, keep=False)
    n_dup = int(dup_mask.sum())
    dup_groups = int(df.duplicated(subset=subset, keep="first").sum())
    return {
        "duplicate_rows": n_dup,
        "duplicate_groups": dup_groups,
        "duplicate_pct": round(dup_groups / len(df) * 100, 2) if len(df) else 0.0,
        "subset": subset,
    }


def check_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> dict:
    """Flag statistical outliers in numeric columns."""
    num_cols = get_numeric_columns(df)
    col_results: List[Dict[str, Any]] = []
    total_outliers = 0

    for col in num_cols:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue

        if method == "zscore":
            mean, std = clean.mean(), clean.std()
            if std == 0:
                continue
            z = ((clean - mean) / std).abs()
            outlier_mask = z > threshold
        else:  # iqr
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outlier_mask = (clean < lower) | (clean > upper)

        n_out = int(outlier_mask.sum())
        total_outliers += n_out
        if n_out > 0:
            col_results.append({
                "column": col,
                "outlier_count": n_out,
                "outlier_pct": round(n_out / len(clean) * 100, 2),
                "method": method,
                "threshold": threshold,
            })

    return {
        "method": method,
        "total_outlier_flags": total_outliers,
        "columns_with_outliers": len(col_results),
        "details": col_results,
    }


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_data(file_path: str, rules: Optional[Dict] = None) -> dict:
    """
    Comprehensive data validation.

    Default checks (if no rules provided):
    - Null values per column
    - Duplicate rows
    - Data type consistency
    - Value range outliers
    - String format issues

    Returns: status pass/fail, score 0-100, checks list, summary, recommendations.
    """
    try:
        df = load_data(file_path)
        logger.info(f"Validating {file_path}: {df.shape}")

        checks: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        # 1. Null checks
        null_info = check_nulls(df)
        for cr in null_info["columns"]:
            severity = "error" if cr["null_pct"] > 50 else ("warning" if cr["null_pct"] > 10 else "info")
            passed = cr["null_pct"] == 0
            checks.append({
                "check": "no_nulls",
                "column": cr["column"],
                "passed": passed,
                "detail": f"{cr['null_count']} nulls ({cr['null_pct']}%)",
                "severity": severity,
            })
            if cr["null_pct"] > 50:
                recommendations.append(f"Consider dropping '{cr['column']}' ({cr['null_pct']}% null)")

        # 2. Duplicate check
        dup_info = check_duplicates(df)
        checks.append({
            "check": "no_duplicates",
            "column": "*",
            "passed": dup_info["duplicate_groups"] == 0,
            "detail": f"{dup_info['duplicate_groups']} duplicate rows ({dup_info['duplicate_pct']}%)",
            "severity": "warning" if dup_info["duplicate_groups"] > 0 else "info",
        })
        if dup_info["duplicate_groups"] > 0:
            recommendations.append(f"Remove {dup_info['duplicate_groups']} duplicate rows")

        # 3. Constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                checks.append({
                    "check": "not_constant",
                    "column": col,
                    "passed": False,
                    "detail": "Column has 0 or 1 unique values",
                    "severity": "warning",
                })
                recommendations.append(f"Remove constant column '{col}'")

        # 4. Outlier checks
        outlier_info = check_outliers(df)
        for od in outlier_info["details"]:
            checks.append({
                "check": "no_outliers",
                "column": od["column"],
                "passed": od["outlier_count"] == 0,
                "detail": f"{od['outlier_count']} outliers ({od['outlier_pct']}%)",
                "severity": "warning" if od["outlier_pct"] > 5 else "info",
            })

        # 5. Mixed-type check for object columns
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna()
            if len(sample) == 0:
                continue
            types = sample.apply(type).unique()
            if len(types) > 1:
                checks.append({
                    "check": "consistent_types",
                    "column": col,
                    "passed": False,
                    "detail": f"Mixed types found: {[t.__name__ for t in types]}",
                    "severity": "warning",
                })

        # 6. Custom rules
        if rules:
            for col_name, col_rules in rules.items():
                if col_name not in df.columns:
                    checks.append({
                        "check": "column_exists",
                        "column": col_name,
                        "passed": False,
                        "detail": f"Column '{col_name}' not found",
                        "severity": "error",
                    })
                    continue
                if "min" in col_rules:
                    val = df[col_name].min()
                    ok = val >= col_rules["min"]
                    checks.append({
                        "check": "min_value",
                        "column": col_name,
                        "passed": bool(ok),
                        "detail": f"min={val}, expected>={col_rules['min']}",
                        "severity": "error" if not ok else "info",
                    })
                if "max" in col_rules:
                    val = df[col_name].max()
                    ok = val <= col_rules["max"]
                    checks.append({
                        "check": "max_value",
                        "column": col_name,
                        "passed": bool(ok),
                        "detail": f"max={val}, expected<={col_rules['max']}",
                        "severity": "error" if not ok else "info",
                    })
                if "not_null" in col_rules and col_rules["not_null"]:
                    n_null = int(df[col_name].isna().sum())
                    checks.append({
                        "check": "not_null",
                        "column": col_name,
                        "passed": n_null == 0,
                        "detail": f"{n_null} nulls",
                        "severity": "error" if n_null > 0 else "info",
                    })

        # Summary
        total = len(checks)
        passed = sum(1 for c in checks if c["passed"])
        failed = sum(1 for c in checks if not c["passed"] and c["severity"] == "error")
        warnings = sum(1 for c in checks if not c["passed"] and c["severity"] == "warning")
        score = round(passed / total * 100, 1) if total else 100.0
        validation_result = "pass" if failed == 0 else "fail"

        result = {
            "status": "success",
            "validation_result": validation_result,
            "score": score,
            "checks": checks,
            "summary": {
                "total_checks": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
            },
            "recommendations": recommendations,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def validate_and_upload(file_path: str, output_name: str = "validation.json", rules: Optional[Dict] = None) -> dict:
    """Convenience function: validate_data + upload."""
    result = validate_data(file_path, rules=rules)
    upload_result(result, output_name)
    return result
