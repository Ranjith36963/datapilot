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


def _save_fig(fig, file_path: str, chart_name: str) -> str:
    """Save matplotlib figure and return path."""
    import matplotlib.pyplot as plt
    out_dir = Path(file_path).parent
    chart_path = str(out_dir / f"{chart_name}.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path


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
                data = df.groupby(x)[y].mean().sort_values(ascending=False).head(20)
                data.plot(kind="bar", ax=ax)
                ax.set_ylabel(y)
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
            plt.close("all")
            return safe_json_serialize({
                "status": "success",
                "chart_type": "pair",
                "chart_path": chart_path,
            })

        else:
            return {"status": "error", "message": f"Unknown chart_type: {chart_type}"}

        ax.set_title(chart_title)
        plt.tight_layout()
        chart_path = _save_fig(fig, file_path, chart_type)

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

        result = {
            "status": "success",
            "chart_type": chart_type,
            "chart_path": chart_path,
            "chart_html_path": html_path,
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
