"""
Charts — create any type of chart.

Uses matplotlib/seaborn for static charts, plotly for interactive HTML.
"""

import io
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.helpers import load_data, setup_logging, get_numeric_columns, get_categorical_columns
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.charts")


def _compute_chart_summary(
    df: pd.DataFrame,
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a small summary of the data actually plotted, for narration.

    Returns a dict with aggregation type and key data points so the LLM
    can narrate real numbers instead of hallucinating.
    """
    summary: Dict[str, Any] = {"x_column": x, "y_column": y, "chart_type": chart_type}
    try:
        if chart_type in ("bar", "barh") and x and y:
            if not pd.api.types.is_numeric_dtype(df[y]):
                # Binary/categorical y — proportion of first sorted value per x group
                target_val = sorted(df[y].dropna().unique())[0]
                grouped = df.groupby(x)[y].apply(lambda s: (s == target_val).mean() * 100)
                data = grouped.sort_values(ascending=False).head(20)
                summary["aggregation"] = "proportion"
                summary["target_value"] = str(target_val)
                summary["data"] = [
                    {str(x): str(k), f"{y} Rate (%)": round(v, 1)} for k, v in data.items()
                ]
            else:
                data = df.groupby(x)[y].mean().sort_values(ascending=False).head(20)
                summary["aggregation"] = "mean"
                summary["data"] = [
                    {str(x): str(k), f"Mean {y}": round(v, 2)} for k, v in data.items()
                ]
        elif chart_type in ("bar", "barh", "count") and x and not y:
            vc = df[x].value_counts().head(20)
            summary["aggregation"] = "count"
            summary["data"] = [
                {str(x): str(k), "Count": int(v)} for k, v in vc.items()
            ]
        elif chart_type == "scatter" and x and y:
            corr = df[[x, y]].dropna().corr().iloc[0, 1]
            summary["aggregation"] = "correlation"
            summary["correlation"] = round(float(corr), 4)
            summary["x_stats"] = {
                "mean": round(float(df[x].mean()), 2),
                "std": round(float(df[x].std()), 2),
            }
            summary["y_stats"] = {
                "mean": round(float(df[y].mean()), 2),
                "std": round(float(df[y].std()), 2),
            }
        elif chart_type == "histogram" and x:
            col = df[x].dropna()
            summary["aggregation"] = "distribution"
            summary["stats"] = {
                "mean": round(float(col.mean()), 2),
                "median": round(float(col.median()), 2),
                "std": round(float(col.std()), 2),
                "min": round(float(col.min()), 2),
                "max": round(float(col.max()), 2),
            }
        elif chart_type == "box" and x and y:
            summary["aggregation"] = "quartiles"
            summary["data"] = []
            for group, sub in df.groupby(x):
                vals = sub[y].dropna()
                if len(vals) > 0:
                    summary["data"].append({
                        str(x): str(group),
                        "median": round(float(vals.median()), 2),
                        "q1": round(float(vals.quantile(0.25)), 2),
                        "q3": round(float(vals.quantile(0.75)), 2),
                    })
            summary["data"] = summary["data"][:20]
        elif chart_type == "pie" and x:
            vc = df[x].value_counts().head(10)
            total = vc.sum()
            summary["aggregation"] = "proportion"
            summary["data"] = [
                {str(x): str(k), "Count": int(v), "Pct": round(v / total * 100, 1)}
                for k, v in vc.items()
            ]
        elif chart_type == "heatmap":
            num_cols = get_numeric_columns(df)
            corr = df[num_cols].corr()
            # Find top 5 strongest off-diagonal correlations
            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    pairs.append({
                        "col1": num_cols[i], "col2": num_cols[j],
                        "correlation": round(float(corr.iloc[i, j]), 4),
                    })
            pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
            summary["aggregation"] = "correlation_matrix"
            summary["top_pairs"] = pairs[:5]
    except Exception as e:
        summary["summary_error"] = str(e)
    return summary


def _save_fig(fig, file_path: str, chart_name: str) -> tuple:
    """Save matplotlib figure and return (path, base64_string)."""
    import matplotlib.pyplot as plt
    out_dir = Path(file_path).parent
    chart_path = str(out_dir / f"{chart_name}.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")

    # Also capture base64 for inline display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    plt.close(fig)
    return chart_path, chart_b64


def create_chart(
    file_path: str,
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Universal chart creation.

    chart_type: bar, barh, line, scatter, histogram, box, violin,
                heatmap, pie, area, density, count, pair.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = load_data(file_path)
        chart_title = title or f"{chart_type.title()} Chart"

        no_x_required = {"heatmap", "pair"}
        if chart_type not in no_x_required and not x:
            return {"status": "error", "message": f"x parameter is required for {chart_type} chart"}

        logger.info(f"Creating {chart_type} chart: x={x}, y={y}")

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            if y:
                if pd.api.types.is_numeric_dtype(df[y]):
                    data = df.groupby(x)[y].mean().sort_values(ascending=False).head(20)
                    data.plot(kind="bar", ax=ax)
                    ax.set_ylabel(f"Mean {y}")
                elif df[y].nunique() <= 2:
                    # Binary categorical y: show proportion of first sorted value per x group
                    target_val = sorted(df[y].dropna().unique())[0]
                    proportions = df.groupby(x)[y].apply(lambda s: (s == target_val).mean() * 100)
                    proportions = proportions.sort_values(ascending=False).head(20)
                    proportions.plot(kind="bar", ax=ax)
                    ax.set_ylabel(f"{y} = {target_val} Rate (%)")
                else:
                    # Multi-class categorical y: use crosstab proportions, stacked bar
                    ct = pd.crosstab(df[x], df[y], normalize="index") * 100
                    ct.head(20).plot(kind="bar", stacked=True, ax=ax)
                    ax.set_ylabel(f"{y} Distribution (%)")
            else:
                df[x].value_counts().head(20).plot(kind="bar", ax=ax)

        elif chart_type == "barh":
            if y:
                data = df.groupby(x)[y].mean().sort_values().tail(20)
                data.plot(kind="barh", ax=ax)
            else:
                df[x].value_counts().tail(20).plot(kind="barh", ax=ax)

        elif chart_type == "line":
            if y:
                df.plot(x=x, y=y, kind="line", ax=ax)
            else:
                df[x].plot(kind="line", ax=ax)

        elif chart_type == "scatter":
            if y:
                sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, alpha=0.6)
            else:
                return {"status": "error", "message": "Scatter plot requires y parameter"}

        elif chart_type == "histogram":
            bins = kwargs.get("bins", 30)
            df[x].dropna().hist(bins=bins, ax=ax, edgecolor="black")
            ax.set_xlabel(x)

        elif chart_type == "box":
            if y:
                sns.boxplot(data=df, x=x, y=y, ax=ax)
            else:
                df[x].dropna().plot(kind="box", ax=ax)

        elif chart_type == "violin":
            if y:
                sns.violinplot(data=df, x=x, y=y, ax=ax)
            else:
                return {"status": "error", "message": "Violin plot requires x and y"}

        elif chart_type == "heatmap":
            num_cols = get_numeric_columns(df)
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=len(num_cols) <= 10, fmt=".2f", cmap="coolwarm", ax=ax)
            chart_title = title or "Correlation Heatmap"

        elif chart_type == "pie":
            vc = df[x].value_counts().head(10)
            vc.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")

        elif chart_type == "area":
            if y:
                df.plot(x=x, y=y, kind="area", ax=ax, alpha=0.5)
            else:
                df[x].plot(kind="area", ax=ax, alpha=0.5)

        elif chart_type == "density":
            sns.kdeplot(data=df, x=x, hue=hue, ax=ax, fill=True)

        elif chart_type == "count":
            sns.countplot(data=df, x=x, hue=hue, ax=ax, order=df[x].value_counts().head(20).index)

        elif chart_type == "pair":
            plt.close(fig)  # close the default figure
            num_cols = get_numeric_columns(df)[:6]
            g = sns.pairplot(df[num_cols].dropna(), diag_kind="kde")
            chart_path = str(Path(file_path).parent / "pairplot.png")
            g.savefig(chart_path, dpi=100, bbox_inches="tight")
            # Capture base64
            buf = io.BytesIO()
            g.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            pair_b64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close("all")
            return safe_json_serialize({
                "status": "success",
                "chart_type": "pair",
                "chart_path": chart_path,
                "chart_base64": pair_b64,
            })

        else:
            return {"status": "error", "message": f"Unknown chart_type: {chart_type}"}

        ax.set_title(chart_title)
        plt.tight_layout()
        chart_path, chart_b64 = _save_fig(fig, file_path, chart_type)

        # Interactive Plotly HTML (optional)
        html_path = None
        try:
            import plotly.express as px
            if chart_type == "scatter" and y:
                pfig = px.scatter(df, x=x, y=y, color=hue, title=chart_title)
            elif chart_type == "bar" and y:
                agg = df.groupby(x)[y].mean().reset_index().sort_values(y, ascending=False).head(20)
                pfig = px.bar(agg, x=x, y=y, title=chart_title)
            elif chart_type == "line" and y:
                pfig = px.line(df, x=x, y=y, title=chart_title)
            elif chart_type == "histogram":
                pfig = px.histogram(df, x=x, title=chart_title)
            elif chart_type == "box" and y:
                pfig = px.box(df, x=x, y=y, title=chart_title)
            else:
                pfig = None

            if pfig:
                html_path = str(Path(file_path).parent / f"{chart_type}_interactive.html")
                pfig.write_html(html_path)
        except (ImportError, Exception):
            pass

        chart_summary = _compute_chart_summary(df, chart_type, x, y, hue)

        result = {
            "status": "success",
            "chart_type": chart_type,
            "chart_path": chart_path,
            "chart_base64": chart_b64,
            "chart_html_path": html_path,
            "chart_summary": chart_summary,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def auto_chart(file_path: str, x: str, y: Optional[str] = None) -> dict:
    """Automatically pick best chart type based on data types."""
    try:
        df = load_data(file_path)
        x_is_num = pd.api.types.is_numeric_dtype(df[x])
        y_is_num = pd.api.types.is_numeric_dtype(df[y]) if y and y in df.columns else False

        if x_is_num and y_is_num:
            chart_type = "scatter"
        elif x_is_num and not y:
            chart_type = "histogram"
        elif not x_is_num and y_is_num:
            chart_type = "box"
        elif not x_is_num and not y:
            chart_type = "count"
        else:
            chart_type = "bar"

        return create_chart(file_path, chart_type, x, y)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_dashboard(file_path: str, charts: Optional[List[Dict]] = None) -> dict:
    """Create multiple charts in a grid layout."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not charts:
            # Auto dashboard
            df = load_data(file_path)
            num_cols = get_numeric_columns(df)[:4]
            cat_cols = get_categorical_columns(df)[:2]

            results = []
            for col in num_cols[:2]:
                r = create_chart(file_path, "histogram", col)
                results.append(r)
            for col in cat_cols[:2]:
                r = create_chart(file_path, "count", col)
                results.append(r)
            if len(num_cols) >= 2:
                r = create_chart(file_path, "scatter", num_cols[0], num_cols[1])
                results.append(r)

            return safe_json_serialize({
                "status": "success",
                "charts": results,
            })
        else:
            results = []
            for spec in charts:
                r = create_chart(file_path, **spec)
                results.append(r)
            return safe_json_serialize({"status": "success", "charts": results})

    except Exception as e:
        return {"status": "error", "message": str(e)}


def chart_and_upload(file_path: str, output_name: str = "chart_metadata.json", **kwargs) -> dict:
    """Convenience function: create_chart + upload."""
    result = create_chart(file_path, **kwargs)
    upload_result(result, output_name)
    return result
