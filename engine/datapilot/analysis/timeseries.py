"""
Time series analysis — trends, seasonality, stationarity, and forecasting.

Uses statsmodels for decomposition and ARIMA, Prophet for forecasting,
ruptures for change-point detection.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..utils.helpers import load_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result

logger = setup_logging("datapilot.timeseries")


# ---------------------------------------------------------------------------
# Frequency detection
# ---------------------------------------------------------------------------

def _detect_freq(dates: pd.Series) -> str:
    """Guess the frequency of a datetime series."""
    diffs = dates.diff().dropna()
    median_days = diffs.dt.days.median()
    if median_days <= 1:
        return "daily"
    if median_days <= 7:
        return "weekly"
    if median_days <= 31:
        return "monthly"
    if median_days <= 92:
        return "quarterly"
    return "yearly"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_time_series(
    file_path: str,
    date_column: str,
    value_column: str,
    freq: str = "auto",
) -> dict:
    """
    Time series analysis: trend, seasonality, stationarity.
    """
    try:
        df = load_data(file_path)
        if date_column not in df.columns or value_column not in df.columns:
            return {"status": "error", "message": f"Column not found: {date_column} or {value_column}"}

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column, value_column]).sort_values(date_column)
        dates = df[date_column]
        values = df[value_column].astype(float)

        detected_freq = freq if freq != "auto" else _detect_freq(dates)
        n = len(values)

        logger.info(f"Analyzing time series: {n} observations, freq={detected_freq}")

        # Trend (linear regression on ordinal index)
        x_ord = np.arange(n)
        slope, intercept, r, p, se = sp_stats.linregress(x_ord, values.values)
        if p < 0.05:
            direction = "increasing" if slope > 0 else "decreasing"
        else:
            direction = "flat"

        # Stationarity (ADF test)
        stationarity: dict[str, Any] = {}
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(values.values, autolag="AIC")
            stationarity = {
                "is_stationary": bool(adf_result[1] < 0.05),
                "adf_statistic": round(float(adf_result[0]), 4),
                "adf_pvalue": round(float(adf_result[1]), 6),
            }
        except ImportError:
            stationarity = {"is_stationary": None, "note": "statsmodels not available"}

        # Seasonality detection
        seasonality: dict[str, Any] = {"detected": False}
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            freq_map = {"daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}
            period = freq_map.get(detected_freq, 7)
            if n >= 2 * period and period > 1:
                decomp = seasonal_decompose(values.values, period=period, model="additive", extrapolate_trend="freq")
                seasonal_strength = float(np.std(decomp.seasonal) / (np.std(decomp.seasonal) + np.std(decomp.resid) + 1e-12))
                seasonality = {
                    "detected": seasonal_strength > 0.1,
                    "period": period,
                    "strength": round(seasonal_strength, 4),
                }
        except (ImportError, Exception):
            pass

        # Summary stats
        summary_stats = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "latest_value": round(float(values.iloc[-1]), 4),
            "change_from_start": round(float(values.iloc[-1] - values.iloc[0]), 4),
            "change_pct": round(float((values.iloc[-1] - values.iloc[0]) / abs(values.iloc[0]) * 100), 2)
            if values.iloc[0] != 0 else None,
        }

        result = {
            "status": "success",
            "date_column": date_column,
            "value_column": value_column,
            "frequency": detected_freq,
            "n_observations": n,
            "date_range": {"start": str(dates.iloc[0]), "end": str(dates.iloc[-1])},
            "trend": {
                "direction": direction,
                "slope": round(float(slope), 6),
                "pvalue": round(float(p), 6),
            },
            "seasonality": seasonality,
            "stationarity": stationarity,
            "summary_stats": summary_stats,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def forecast(
    file_path: str,
    date_column: str,
    value_column: str,
    periods: int = 30,
    method: str = "auto",
) -> dict:
    """
    Forecast future values.

    Methods: auto, prophet, arima, exponential_smoothing, naive.
    """
    try:
        df = load_data(file_path)
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column, value_column]).sort_values(date_column)
        dates = df[date_column]
        values = df[value_column].astype(float)
        n = len(values)

        detected_freq = _detect_freq(dates)
        logger.info(f"Forecasting {periods} periods via {method}")

        # Choose method
        chosen = method
        if method == "auto":
            try:
                import prophet  # noqa: F401
                chosen = "prophet"
            except ImportError:
                try:
                    import statsmodels  # noqa: F401
                    chosen = "exponential_smoothing"
                except ImportError:
                    chosen = "naive"

        forecast_list: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        if chosen == "prophet":
            try:
                from prophet import Prophet
                pdf = pd.DataFrame({"ds": dates.values, "y": values.values})
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                m.fit(pdf)

                freq_code = {"daily": "D", "weekly": "W", "monthly": "MS", "quarterly": "QS", "yearly": "YS"}
                future = m.make_future_dataframe(periods=periods, freq=freq_code.get(detected_freq, "D"))
                fc = m.predict(future)
                fc_future = fc.tail(periods)

                for _, row in fc_future.iterrows():
                    forecast_list.append({
                        "date": str(row["ds"].date()),
                        "predicted": round(float(row["yhat"]), 4),
                        "lower_bound": round(float(row["yhat_lower"]), 4),
                        "upper_bound": round(float(row["yhat_upper"]), 4),
                    })

                # In-sample error
                in_sample = fc.head(n)
                mape = float(np.mean(np.abs((values.values - in_sample["yhat"].values) / (values.values + 1e-12))) * 100)
                rmse = float(np.sqrt(np.mean((values.values - in_sample["yhat"].values) ** 2)))
                metrics = {"mape": round(mape, 2), "rmse": round(rmse, 4)}

            except ImportError:
                return {"status": "error", "message": "Prophet not installed"}

        elif chosen == "arima":
            try:
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(values.values, order=(1, 1, 1))
                fit = model.fit()
                fc = fit.forecast(steps=periods)

                last_date = dates.iloc[-1]
                freq_code = {"daily": "D", "weekly": "W", "monthly": "MS", "quarterly": "QS", "yearly": "YS"}
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq_code.get(detected_freq, "D"))[1:]

                for i in range(periods):
                    forecast_list.append({
                        "date": str(future_dates[i].date()),
                        "predicted": round(float(fc[i]), 4),
                        "lower_bound": None,
                        "upper_bound": None,
                    })
            except ImportError:
                return {"status": "error", "message": "statsmodels not installed"}

        elif chosen == "exponential_smoothing":
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                model = ExponentialSmoothing(values.values, trend="add", seasonal=None)
                fit = model.fit()
                fc = fit.forecast(periods)

                last_date = dates.iloc[-1]
                freq_code = {"daily": "D", "weekly": "W", "monthly": "MS", "quarterly": "QS", "yearly": "YS"}
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq_code.get(detected_freq, "D"))[1:]

                for i in range(periods):
                    forecast_list.append({
                        "date": str(future_dates[i].date()),
                        "predicted": round(float(fc[i]), 4),
                        "lower_bound": None,
                        "upper_bound": None,
                    })
            except ImportError:
                return {"status": "error", "message": "statsmodels not installed"}

        else:  # naive
            last_val = float(values.iloc[-1])
            last_date = dates.iloc[-1]
            freq_code = {"daily": "D", "weekly": "W", "monthly": "MS", "quarterly": "QS", "yearly": "YS"}
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq_code.get(detected_freq, "D"))[1:]
            for i in range(periods):
                forecast_list.append({
                    "date": str(future_dates[i].date()),
                    "predicted": round(last_val, 4),
                    "lower_bound": None,
                    "upper_bound": None,
                })

        result = {
            "status": "success",
            "method": chosen,
            "periods_forecast": periods,
            "forecast": forecast_list,
            "metrics": metrics,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def detect_change_points(file_path: str, date_column: str, value_column: str) -> dict:
    """Find where the trend significantly changed."""
    try:
        df = load_data(file_path)
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column, value_column]).sort_values(date_column)
        values = df[value_column].astype(float).values

        change_points: list[dict[str, Any]] = []

        try:
            import ruptures as rpt
            algo = rpt.Pelt(model="rbf").fit(values)
            bkps = algo.predict(pen=10)
            for bp in bkps[:-1]:  # last is always len(values)
                if 0 < bp < len(values):
                    change_points.append({
                        "index": int(bp),
                        "date": str(df[date_column].iloc[bp]),
                        "value_before": round(float(values[max(0, bp - 5):bp].mean()), 4),
                        "value_after": round(float(values[bp:min(len(values), bp + 5)].mean()), 4),
                    })
        except ImportError:
            # Fallback: rolling z-score
            window = max(10, len(values) // 20)
            roll_mean = pd.Series(values).rolling(window).mean()
            roll_std = pd.Series(values).rolling(window).std()
            z = ((pd.Series(values) - roll_mean) / (roll_std + 1e-12)).abs()
            peaks = z[z > 3].index.tolist()
            for idx in peaks[:20]:
                change_points.append({
                    "index": int(idx),
                    "date": str(df[date_column].iloc[idx]),
                    "z_score": round(float(z.iloc[idx]), 2),
                })

        return safe_json_serialize({
            "status": "success",
            "n_change_points": len(change_points),
            "change_points": change_points,
        })

    except Exception as e:
        return {"status": "error", "message": str(e)}


def timeseries_and_upload(file_path: str, output_name: str = "timeseries.json", **kwargs) -> dict:
    """Convenience function: analyze_time_series + upload."""
    result = analyze_time_series(file_path, **kwargs)
    upload_result(result, output_name)
    return result
