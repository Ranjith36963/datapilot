"""
Survival analysis — model time-to-event (when will customer churn?).

Uses lifelines for Kaplan-Meier curves and Cox Proportional Hazards.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from veritly_analysis.utils import (
    load_data,
    safe_json_serialize,
    setup_logging,
    upload_result,
)


logger = setup_logging("survival_analysis")


def survival_analysis(
    file_path: str,
    duration_column: str,
    event_column: str,
    group_column: Optional[str] = None,
) -> dict:
    """
    Kaplan-Meier survival analysis.

    duration_column: time until event (e.g., months as customer).
    event_column: 1 = event happened, 0 = censored.
    group_column: optional grouping for comparison.
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        df = load_data(file_path)
        for col in [duration_column, event_column]:
            if col not in df.columns:
                return {"status": "error", "message": f"Column '{col}' not found"}

        df = df.dropna(subset=[duration_column, event_column])
        T = df[duration_column].astype(float)
        E = df[event_column].astype(int)

        n_subjects = len(T)
        n_events = int(E.sum())
        censoring_rate = round(1 - n_events / n_subjects, 4) if n_subjects else 0.0

        logger.info(f"Survival analysis: {n_subjects} subjects, {n_events} events")

        # Fit KM curve
        kmf = KaplanMeierFitter()
        kmf.fit(T, E)
        median_survival = float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None

        # Survival curve data
        timeline = kmf.survival_function_.index.tolist()
        surv_prob = kmf.survival_function_.iloc[:, 0].tolist()
        ci = kmf.confidence_interval_survival_function_
        ci_lower = ci.iloc[:, 0].tolist()
        ci_upper = ci.iloc[:, 1].tolist()

        curve = []
        for i in range(len(timeline)):
            curve.append({
                "time": round(float(timeline[i]), 4),
                "survival_prob": round(float(surv_prob[i]), 4),
                "ci_lower": round(float(ci_lower[i]), 4),
                "ci_upper": round(float(ci_upper[i]), 4),
            })

        # Survival at standard times
        survival_at = {}
        for t_label, t_val in [("30_days", 30), ("90_days", 90), ("180_days", 180), ("365_days", 365)]:
            try:
                prob = float(kmf.predict(t_val))
                survival_at[t_label] = round(prob, 4)
            except Exception:
                pass

        # Group comparison
        group_comparison: Dict[str, Any] = {}
        if group_column and group_column in df.columns:
            groups = df[group_column].dropna().unique()
            group_info = []
            for g in groups:
                mask = df[group_column] == g
                gT = T[mask]
                gE = E[mask]
                gkmf = KaplanMeierFitter()
                gkmf.fit(gT, gE)
                gmed = float(gkmf.median_survival_time_) if not np.isinf(gkmf.median_survival_time_) else None
                group_info.append({
                    "group": str(g),
                    "median_survival": gmed,
                    "n_subjects": int(mask.sum()),
                })

            # Log-rank test (first two groups)
            if len(groups) >= 2:
                g1_mask = df[group_column] == groups[0]
                g2_mask = df[group_column] == groups[1]
                lr = logrank_test(T[g1_mask], T[g2_mask], E[g1_mask], E[g2_mask])
                group_comparison = {
                    "logrank_statistic": round(float(lr.test_statistic), 4),
                    "logrank_pvalue": round(float(lr.p_value), 6),
                    "groups": group_info,
                }
            else:
                group_comparison = {"groups": group_info}

        result = {
            "status": "success",
            "n_subjects": n_subjects,
            "n_events": n_events,
            "censoring_rate": censoring_rate,
            "median_survival": median_survival,
            "survival_curve": curve[:200],  # cap for JSON
            "survival_at_times": survival_at,
            "group_comparison": group_comparison,
        }
        return safe_json_serialize(result)

    except ImportError:
        return {"status": "error", "message": "lifelines not installed. Run: pip install lifelines"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def cox_regression(
    file_path: str,
    duration_column: str,
    event_column: str,
    features: Optional[List[str]] = None,
) -> dict:
    """
    Cox Proportional Hazards regression.

    Returns hazard ratios, confidence intervals, and concordance index.
    """
    try:
        from lifelines import CoxPHFitter

        df = load_data(file_path)
        for col in [duration_column, event_column]:
            if col not in df.columns:
                return {"status": "error", "message": f"Column '{col}' not found"}

        cols = features if features else [c for c in df.columns if c not in (duration_column, event_column)]
        # Keep only numeric
        use_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        analysis_df = df[use_cols + [duration_column, event_column]].dropna()

        logger.info(f"Cox regression: {len(analysis_df)} subjects, {len(use_cols)} covariates")

        cph = CoxPHFitter()
        cph.fit(analysis_df, duration_col=duration_column, event_col=event_column)

        summary = cph.summary
        hazard_ratios = []
        for feat in use_cols:
            if feat in summary.index:
                row = summary.loc[feat]
                hr = float(row["exp(coef)"])
                hr_lo = float(row["exp(coef) lower 95%"])
                hr_hi = float(row["exp(coef) upper 95%"])
                p = float(row["p"])

                if hr > 1:
                    interp = f"{hr:.1f}x more likely per unit increase"
                else:
                    interp = f"{1/hr:.1f}x less likely per unit increase"

                hazard_ratios.append({
                    "feature": feat,
                    "hazard_ratio": round(hr, 4),
                    "ci_lower": round(hr_lo, 4),
                    "ci_upper": round(hr_hi, 4),
                    "pvalue": round(p, 6),
                    "interpretation": interp,
                })

        hazard_ratios.sort(key=lambda x: abs(x["hazard_ratio"] - 1), reverse=True)
        concordance = round(float(cph.concordance_index_), 4)

        result = {
            "status": "success",
            "hazard_ratios": hazard_ratios,
            "concordance_index": concordance,
        }
        return safe_json_serialize(result)

    except ImportError:
        return {"status": "error", "message": "lifelines not installed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def survival_and_upload(file_path: str, output_name: str = "survival.json", **kwargs) -> dict:
    """Convenience function: survival_analysis + upload."""
    result = survival_analysis(file_path, **kwargs)
    upload_result(result, output_name)
    return result
