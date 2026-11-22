"""
PharmaPulse - Visualization Utility Module (Dark Theme)
Provides clean, consistent, high-fidelity dark-mode Plotly chart generators matching
a modern enterprise healthcare/pharmaceutical command center aesthetic.

Color Palette:
- Brand Sky: #38BDF8
- Clinical Emerald: #34D399
- Indigo: #818CF8
- Warning Amber: #FBBF24
- Critical Rose: #F87171
- Muted Slate: #94A3B8
- Deep Charcoal: #0B0F19
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

THEME_COLORS = {
    "primary": "#38BDF8",
    "dark": "#0B0F19",
    "surface": "#111827",
    "emerald": "#34D399",
    "rose": "#F87171",
    "amber": "#FBBF24",
    "indigo": "#818CF8",
    "slate": "#94A3B8",
    "light": "#F8FAFC"
}

PALETTE_SEQUENCE = ["#38BDF8", "#34D399", "#818CF8", "#FBBF24", "#F87171", "#94A3B8"]

def apply_modern_layout(fig, height=380, title=None):
    """
    Applies consistent dark enterprise styling across all Plotly figures:
    transparent background, subtle dark gridlines, Plus Jakarta Sans typography,
    and dark hovercards.
    """
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=title if title else fig.layout.title.text if fig.layout.title else None,
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=15, color="#F8FAFC", weight=700)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=35, r=35, t=50, b=35),
        height=height,
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_color="#FFFFFF",
            font_family="Plus Jakarta Sans, Inter, sans-serif",
            bordercolor="rgba(255,255,255,0.15)"
        )
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.15)",
        tickfont=dict(color="#94A3B8", size=11)
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.15)",
        tickfont=dict(color="#94A3B8", size=11)
    )
    return fig

def create_demand_forecast_chart(historical_df, forecast_df, medicine_name, horizon_days=30):
    """
    Plots historical demand series along with the model forecast curve and uncertainty intervals in dark theme.
    """
    fig = go.Figure()

    # Historical demand line (slate)
    fig.add_trace(go.Scatter(
        x=historical_df["date"],
        y=historical_df["actual_demand"],
        mode="lines",
        name="Historical Demand",
        line=dict(color="#94A3B8", width=1.8),
        hovertemplate="<b>Date:</b> %{x}<br><b>Actual:</b> %{y} units<extra></extra>"
    ))

    if not forecast_df.empty:
        # Uncertainty intervals (if available)
        if "lower_bound" in forecast_df.columns and "upper_bound" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
                y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="80% Uncertainty Band"
            ))

        # Forecast line (glowing cyan)
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["predicted_demand"],
            mode="lines+markers",
            name=f"Forecast ({horizon_days}d)",
            line=dict(color="#38BDF8", width=2.6),
            marker=dict(size=4, color="#38BDF8"),
            hovertemplate="<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:.1f} units<extra></extra>"
        ))

    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        )
    )
    apply_modern_layout(fig, height=420, title=f"Demand Trajectory & Forward Forecast — {medicine_name}")
    fig.update_xaxes(title="Timeline")
    fig.update_yaxes(title="Units Demanded")
    return fig

def create_confusion_matrix_heatmap(cm, class_names, title="Confusion Matrix"):
    """
    Plots an annotated confusion matrix heatmap with dark blue/cyan hues.
    """
    cm_array = np.array(cm)
    fig = px.imshow(
        cm_array,
        labels=dict(x="Predicted Class", y="Actual Truth", color="Observations"),
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale=[[0, "#0F172A"], [0.5, "#0284C7"], [1.0, "#38BDF8"]]
    )
    apply_modern_layout(fig, height=340, title=title)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def create_model_comparison_bar(eval_df, metric_name, title=None):
    """
    Plots a comparative horizontal bar chart across ML algorithms in dark mode.
    """
    df_plot = eval_df[eval_df["metric_name"] == metric_name].sort_values("metric_value", ascending=True)
    if title is None:
        title = f"Algorithm Benchmark: {metric_name}"

    fig = px.bar(
        df_plot,
        x="metric_value",
        y="model_name",
        orientation="h",
        color="model_name",
        color_discrete_sequence=PALETTE_SEQUENCE,
        text="metric_value"
    )
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)"))
    )
    apply_modern_layout(fig, height=300, title=title)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=metric_name)
    fig.update_yaxes(title="Model")
    return fig

def create_cluster_scatter(df_clusters, x_col, y_col, z_col=None):
    """
    Plots 2D or 3D scatter plots of medicine behavioural clusters with vibrant neon palette.
    """
    if z_col is not None and z_col in df_clusters.columns:
        fig = px.scatter_3d(
            df_clusters,
            x=x_col,
            y=y_col,
            z=z_col,
            color="cluster_label",
            hover_name="medicine_name",
            color_discrete_sequence=["#38BDF8", "#34D399", "#F87171", "#FBBF24"],
            title="3D PCA Feature Space of SKU Segments"
        )
        fig.update_layout(
            template="plotly_dark",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)"
        )
    else:
        fig = px.scatter(
            df_clusters,
            x=x_col,
            y=y_col,
            color="cluster_label",
            hover_name="medicine_name",
            size="total_units_sold" if "total_units_sold" in df_clusters.columns else None,
            color_discrete_sequence=["#38BDF8", "#34D399", "#F87171", "#FBBF24"],
            title="2D SKU Behavioral Clustering Map"
        )
        apply_modern_layout(fig, height=400)
    return fig

def create_gauge_chart(value, title, max_val=1.0, threshold_warn=0.5, threshold_crit=0.75):
    """
    Creates an informative dark gauge indicator for risk index.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"family": "Plus Jakarta Sans, sans-serif", "size": 32, "color": "#F8FAFC"}, "valueformat": ".1%"},
        title={"text": title, "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 14, "color": "#94A3B8", "weight": 700}},
        gauge={
            "axis": {"range": [0, max_val], "tickformat": ".0%", "tickcolor": "#94A3B8"},
            "bar": {"color": "#38BDF8", "thickness": 0.28},
            "bgcolor": "#1E293B",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, threshold_warn], "color": "rgba(52, 211, 153, 0.18)"},
                {"range": [threshold_warn, threshold_crit], "color": "rgba(245, 158, 11, 0.22)"},
                {"range": [threshold_crit, max_val], "color": "rgba(239, 68, 68, 0.25)"}
            ],
            "threshold": {
                "line": {"color": "#F87171", "width": 4},
                "thickness": 0.75,
                "value": threshold_crit
            }
        }
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif"),
        height=250,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig
