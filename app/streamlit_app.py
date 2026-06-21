"""Final Streamlit dashboard for the BVC portfolio optimization workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import optimization, scoring, simulation
from src.optimization import compute_investment_amounts, optimize_portfolio
from src.scoring import score_latest_stocks
from src.simulation import historical_bootstrap_monte_carlo
from src.utils import is_valid_file


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
RETURNS_PATH = PROJECT_ROOT / "data" / "processed" / "returns.csv"
SCORING_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "scoring_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
RAW_PRICES_PATH = PROJECT_ROOT / "data" / "raw" / "raw_prices.csv"

OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
LATEST_SCORES_PATH = OUTPUTS_DIR / "latest_scores.csv"
scoring.LATEST_SCORES_OUTPUT_PATH = LATEST_SCORES_PATH
optimization.ALLOCATION_OUTPUT_PATH = OUTPUTS_DIR / "optimal_allocation.csv"
optimization.INVESTMENT_AMOUNTS_OUTPUT_PATH = OUTPUTS_DIR / "investment_amounts.csv"
simulation.SIMULATION_PATHS_OUTPUT_PATH = OUTPUTS_DIR / "simulation_paths.csv"
simulation.SIMULATION_SUMMARY_OUTPUT_PATH = OUTPUTS_DIR / "simulation_summary.csv"

ACTIVE_WEIGHT_THRESHOLD = 1e-4
WEAK_ROC_AUC_THRESHOLD = 0.55
WEAK_F1_THRESHOLD = 0.10
CONCENTRATION_WEIGHT_THRESHOLD = 0.35
CONCENTRATION_EFFECTIVE_ASSETS_THRESHOLD = 5.0


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f8fb;
            --panel: #ffffff;
            --text: #0f172a;
            --muted: #64748b;
            --border: #d8e0ea;
            --blue: #1d4ed8;
            --navy: #0f2747;
            --green: #047857;
            --red: #b91c1c;
            --amber: #b45309;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #edf2f7 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }

        .dashboard-hero {
            background: linear-gradient(135deg, #0f2747 0%, #1d4ed8 100%);
            color: #ffffff;
            padding: 28px 30px;
            border-radius: 14px;
            margin-bottom: 22px;
            box-shadow: 0 14px 35px rgba(15, 39, 71, 0.14);
        }

        .dashboard-hero h1 {
            font-size: 34px;
            line-height: 1.18;
            margin: 0 0 8px 0;
            color: #ffffff;
        }

        .dashboard-hero p {
            margin: 0;
            color: #dbeafe;
            font-size: 16px;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px 18px 16px 18px;
            min-height: 118px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .metric-label {
            color: var(--muted);
            font-size: 13px;
            font-weight: 650;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .metric-value {
            color: var(--text);
            font-size: 26px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .metric-help {
            color: var(--muted);
            font-size: 13px;
        }

        .info-panel, .warning-panel, .method-panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px 20px;
            margin: 12px 0 18px 0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }

        .warning-panel {
            border-left: 5px solid var(--amber);
            background: #fffbeb;
        }

        .method-panel {
            border-left: 5px solid var(--blue);
        }

        .section-title {
            font-size: 20px;
            font-weight: 800;
            color: var(--navy);
            margin: 6px 0 6px 0;
        }

        .small-muted {
            color: var(--muted);
            font-size: 14px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        .stButton button {
            border-radius: 10px;
            font-weight: 700;
            padding: 0.65rem 1rem;
        }

        .stButton button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, var(--navy), var(--blue));
            border: 1px solid var(--blue);
            color: #ffffff;
        }

        .stButton button[data-testid="stBaseButton-primary"]:hover {
            border-color: var(--navy);
            color: #ffffff;
            filter: brightness(0.98);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--blue);
            border-bottom-color: var(--blue);
        }

        div[data-baseweb="tab-highlight"] {
            background-color: var(--blue);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _format_money(value: float) -> str:
    return f"{value:,.2f} MAD"


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2%}"


def _format_number(value: float | int) -> str:
    return f"{value:,.0f}"


def _metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="dashboard-hero">
            <h1>Optimisation intelligente d’un portefeuille d’investissement</h1>
            <p>
                Tableau de bord final pour le scoring des actions, l’allocation optimale
                sous contraintes et la simulation Monte Carlo par bootstrap historique.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_missing_files(missing_paths: list[Path]) -> None:
    missing_items = "".join(
        f"<li><code>{_relative_path(path)}</code></li>" for path in missing_paths
    )
    st.markdown(
        f"""
        <div class="warning-panel">
            <div class="section-title">Pipeline requis</div>
            <p><strong>Veuillez d’abord exécuter le pipeline : python scripts/run_pipeline.py</strong></p>
            <p class="small-muted">Fichiers manquants ou vides :</p>
            <ul>{missing_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _check_required_files() -> bool:
    required_paths = [FEATURES_PATH, RETURNS_PATH, MODEL_PATH, SCALER_PATH]
    missing_paths = [path for path in required_paths if not is_valid_file(path)]
    if missing_paths:
        _show_missing_files(missing_paths)
        return False
    return True


def _load_features(path: Path) -> pd.DataFrame:
    features = pd.read_csv(path)
    if "date" in features.columns:
        features["date"] = pd.to_datetime(features["date"], errors="coerce")
    return features


def _load_returns(path: Path) -> pd.DataFrame:
    returns = pd.read_csv(path)
    if "date" in returns.columns:
        returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
        returns = returns.set_index("date")
    return returns


def _count_tickers_in_csv(path: Path) -> int | None:
    if not is_valid_file(path):
        return None

    try:
        ticker_df = pd.read_csv(path, usecols=["ticker"])
    except Exception:
        return None

    return int(ticker_df["ticker"].dropna().astype(str).nunique())


def _load_pickle(path: Path):
    if not is_valid_file(path):
        raise RuntimeError(
            f"`{_relative_path(path)}` is missing or empty. "
            "Veuillez d’abord exécuter le pipeline : python scripts/run_pipeline.py"
        )

    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load `{_relative_path(path)}`. "
            "Veuillez d’abord exécuter le pipeline : python scripts/run_pipeline.py"
        ) from exc


def _helpful_recommendation_error(exc: Exception, risk_profile: str) -> str:
    message = str(exc)
    if "infeasible for the number of aligned assets" in message:
        return (
            f"Le profil `{risk_profile}` est infaisable avec les titres alignés. "
            "Collectez et scorez plus d’actions, ou choisissez un profil moins restrictif. "
            f"Détails : {message}"
        )
    if "missing or empty" in message or "Could not load" in message:
        return f"{message} Veuillez d’abord exécuter le pipeline : python scripts/run_pipeline.py"
    return message


def _warn_if_three_ticker_scores(scores_df: pd.DataFrame, source: str) -> None:
    raw_ticker_count = _count_tickers_in_csv(RAW_PRICES_PATH)
    score_ticker_count = int(scores_df["ticker"].dropna().astype(str).nunique())
    if raw_ticker_count and raw_ticker_count > 3 and score_ticker_count == 3:
        st.warning(
            f"{source} contient seulement 3 tickers alors que raw_prices.csv en contient "
            f"{raw_ticker_count}. Veuillez d’abord exécuter le pipeline : "
            "python scripts/run_pipeline.py"
        )


def _warn_if_latest_scores_file_is_stale() -> None:
    if not is_valid_file(LATEST_SCORES_PATH):
        return

    try:
        latest_scores = pd.read_csv(LATEST_SCORES_PATH)
    except Exception:
        return

    if "ticker" in latest_scores.columns:
        _warn_if_three_ticker_scores(latest_scores, "latest_scores.csv")


def _align_scores_with_returns(scores_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    returns_tickers = set(map(str, returns_df.columns))
    aligned_scores = scores_df[
        scores_df["ticker"].astype(str).isin(returns_tickers)
    ].copy()

    if aligned_scores.empty:
        raise ValueError("Aucune action scorée n’est disponible dans returns.csv.")

    dropped_tickers = sorted(set(scores_df["ticker"].astype(str)) - returns_tickers)
    if dropped_tickers:
        st.warning(
            "Certaines actions scorées ne sont pas présentes dans returns.csv et sont "
            f"exclues de l’optimisation : {', '.join(dropped_tickers)}"
        )

    return aligned_scores.reset_index(drop=True)


def _validated_allocation(allocation_df: pd.DataFrame, require_score: bool = True) -> pd.DataFrame:
    required_columns = ["ticker", "weight"]
    if require_score:
        required_columns.append("score")

    missing_columns = [
        column for column in required_columns if column not in allocation_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing allocation columns: {', '.join(missing_columns)}.")

    allocation = allocation_df[required_columns].copy()
    allocation["ticker"] = allocation["ticker"].astype(str).str.strip()
    allocation["weight"] = pd.to_numeric(allocation["weight"], errors="coerce")
    if require_score:
        allocation["score"] = pd.to_numeric(allocation["score"], errors="coerce")

    allocation = allocation.dropna(subset=required_columns)
    allocation = allocation[(allocation["ticker"] != "") & (allocation["weight"] > 0)]
    if allocation.empty:
        raise ValueError("No positive allocation weights are available.")

    return allocation.sort_values("weight", ascending=False).reset_index(drop=True)


def _display_allocation(allocation_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    allocation = _validated_allocation(allocation_df, require_score=True)
    active = allocation[allocation["weight"] > ACTIVE_WEIGHT_THRESHOLD].copy()

    if active.empty:
        active = allocation.nlargest(1, "weight").copy()

    small_weight = float(allocation[allocation["weight"] <= ACTIVE_WEIGHT_THRESHOLD]["weight"].sum())
    active = active.sort_values("weight", ascending=False).reset_index(drop=True)
    chart_rows = active.head(top_n).copy()
    inactive_weight = float(active.iloc[top_n:]["weight"].sum()) + small_weight
    if inactive_weight > ACTIVE_WEIGHT_THRESHOLD:
        chart_rows = pd.concat(
            [
                chart_rows,
                pd.DataFrame(
                    {
                        "ticker": ["Autres"],
                        "weight": [inactive_weight],
                        "score": [np.nan],
                    }
                ),
            ],
            ignore_index=True,
        )

    return chart_rows.sort_values("weight", ascending=False).reset_index(drop=True)


def _portfolio_expected_score(allocation_df: pd.DataFrame) -> float:
    allocation = _validated_allocation(allocation_df, require_score=True)
    weights = allocation["weight"].astype(float)
    scores = allocation["score"].astype(float)
    return float((weights * scores).sum())


def _calculate_portfolio_risk(returns_df: pd.DataFrame, allocation_df: pd.DataFrame) -> dict:
    allocation = _validated_allocation(allocation_df, require_score=False)
    tickers = [
        ticker
        for ticker in allocation["ticker"].astype(str)
        if ticker in returns_df.columns
    ]
    if not tickers:
        raise ValueError("Aucun ticker du portefeuille n’est disponible dans returns.csv.")

    weights = allocation.set_index("ticker").loc[tickers, "weight"].astype(float)
    weights = weights / weights.sum()
    aligned_returns = returns_df[tickers].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if aligned_returns.empty:
        raise ValueError("Aucun historique complet de rendements n’est disponible.")

    portfolio_returns = aligned_returns.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
    daily_volatility = float(pd.Series(portfolio_returns).std())
    annualized_volatility = daily_volatility * (252**0.5)
    variance = float(pd.Series(portfolio_returns).var())
    return {
        "daily_volatility": daily_volatility,
        "annualized_volatility": annualized_volatility,
        "variance": variance,
        "observations": int(len(aligned_returns)),
        "start_date": aligned_returns.index.min(),
        "end_date": aligned_returns.index.max(),
    }


def _risk_contribution(returns_df: pd.DataFrame, allocation_df: pd.DataFrame) -> pd.DataFrame:
    allocation = _validated_allocation(allocation_df, require_score=False)
    tickers = [
        ticker
        for ticker in allocation["ticker"].astype(str)
        if ticker in returns_df.columns
    ]
    if len(tickers) < 2:
        return pd.DataFrame()

    weights = allocation.set_index("ticker").loc[tickers, "weight"].astype(float)
    weights = weights / weights.sum()
    aligned_returns = returns_df[tickers].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if aligned_returns.empty:
        return pd.DataFrame()

    covariance = aligned_returns.cov().to_numpy(dtype=float)
    weight_values = weights.to_numpy(dtype=float)
    portfolio_variance = float(weight_values.T @ covariance @ weight_values)
    if portfolio_variance <= 0 or not np.isfinite(portfolio_variance):
        return pd.DataFrame()

    marginal = covariance @ weight_values
    contributions = weight_values * marginal / portfolio_variance
    return pd.DataFrame(
        {
            "ticker": tickers,
            "risk_contribution": contributions,
        }
    ).sort_values("risk_contribution", ascending=False)


def _target_input(budget: float) -> tuple[float, str]:
    target_mode = st.sidebar.radio(
        "Type d’objectif",
        ["Rendement cible", "Valeur cible"],
        horizontal=True,
    )
    if target_mode == "Rendement cible":
        target_return = st.sidebar.number_input(
            "Rendement cible (%)",
            min_value=-99.0,
            max_value=1000.0,
            value=5.0,
            step=0.5,
        )
        target_value = budget * (1 + target_return / 100)
        return target_value, f"{target_return:.2f}%"

    target_value = st.sidebar.number_input(
        "Valeur cible (MAD)",
        min_value=1.0,
        value=float(round(budget * 1.05, 2)),
        step=100.0,
    )
    return target_value, _format_money(target_value)


def _compute_model_metrics(model, scaler, test_size: float = 0.2) -> dict | None:
    if not is_valid_file(SCORING_DATASET_PATH):
        return None

    try:
        dataset = pd.read_csv(SCORING_DATASET_PATH)
        feature_columns = scoring.get_feature_columns()
        required_columns = ["date", "ticker", "target", *feature_columns]
        if any(column not in dataset.columns for column in required_columns):
            return None

        dataset = dataset[required_columns].copy()
        dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
        dataset["target"] = pd.to_numeric(dataset["target"], errors="coerce")
        for column in feature_columns:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
        dataset = dataset.dropna(subset=required_columns)
        dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

        split_index = int(len(dataset) * (1 - test_size))
        if split_index <= 0 or split_index >= len(dataset):
            return None

        test_df = dataset.iloc[split_index:].copy()
        y_true = test_df["target"].astype(int)
        if y_true.nunique() < 1:
            return None

        x_test = test_df[feature_columns].astype(float)
        y_pred = model.predict(scaler.transform(x_test))
        y_proba = model.predict_proba(scaler.transform(x_test))[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": None,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        }
        if y_true.nunique() == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        return metrics
    except Exception:
        return None


def _model_is_weak(metrics: dict | None) -> bool:
    if not metrics:
        return False
    roc_auc = metrics.get("roc_auc")
    f1_score_value = metrics.get("f1_score")
    return (
        roc_auc is not None
        and roc_auc < WEAK_ROC_AUC_THRESHOLD
    ) or (
        f1_score_value is not None
        and f1_score_value < WEAK_F1_THRESHOLD
    )


def _allocation_diagnostics(allocation_df: pd.DataFrame) -> dict:
    allocation = _validated_allocation(allocation_df, require_score=True)
    weights = allocation["weight"].astype(float)
    weights = weights / weights.sum()
    max_weight = float(weights.max())
    herfindahl = float((weights**2).sum())
    effective_assets = float(1 / herfindahl) if herfindahl > 0 else 0.0
    return {
        "active_count": int((weights > ACTIVE_WEIGHT_THRESHOLD).sum()),
        "max_weight": max_weight,
        "effective_assets": effective_assets,
    }


def _render_warning_panels(
    allocation_df: pd.DataFrame,
    model_metrics: dict | None,
) -> None:
    diagnostics = _allocation_diagnostics(allocation_df)
    warnings = []

    if _model_is_weak(model_metrics):
        warnings.append(
            "Les métriques du modèle semblent faibles. Interprétez les scores comme "
            "un signal d’aide à la décision, pas comme une prévision certaine."
        )
    if diagnostics["active_count"] < 5:
        warnings.append(
            f"Moins de 5 actifs ont un poids significatif "
            f"({diagnostics['active_count']} actifs). Le portefeuille est peu diversifié."
        )
    if (
        diagnostics["max_weight"] >= CONCENTRATION_WEIGHT_THRESHOLD
        or diagnostics["effective_assets"] < CONCENTRATION_EFFECTIVE_ASSETS_THRESHOLD
    ):
        warnings.append(
            "L’allocation est concentrée. Vérifiez que ce niveau de concentration est "
            "compatible avec le profil de risque sélectionné."
        )

    for warning in warnings:
        st.markdown(
            f"""
            <div class="warning-panel">
                <strong>Point d’attention.</strong> {warning}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _plot_scores(scores_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top_scores = scores_df.sort_values("score", ascending=False).head(top_n)
    top_scores = top_scores.sort_values("score")
    fig = px.bar(
        top_scores,
        x="score",
        y="ticker",
        orientation="h",
        text=top_scores["score"].map(lambda value: f"{value:.3f}"),
        color="score",
        color_continuous_scale=["#dbeafe", "#1d4ed8"],
        labels={"score": "Score", "ticker": "Action"},
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": f"Top {len(top_scores)} des actions par score", "x": 0.02},
        height=460,
        coloraxis_showscale=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis_tickformat=".0%",
        autosize=True,
    )
    fig.update_traces(textposition="outside")
    return fig


def _plot_allocation_bar(allocation_df: pd.DataFrame) -> go.Figure:
    active_allocation = _display_allocation(allocation_df).sort_values("weight")
    fig = px.bar(
        active_allocation,
        x="weight",
        y="ticker",
        orientation="h",
        text=active_allocation["weight"].map(lambda value: f"{value:.1%}"),
        color="weight",
        color_continuous_scale=["#dcfce7", "#047857"],
        labels={"weight": "Poids", "ticker": "Action"},
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": "Poids de l'allocation optimale", "x": 0.02},
        height=max(360, 32 * len(active_allocation)),
        coloraxis_showscale=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis_tickformat=".0%",
        autosize=True,
    )
    fig.update_traces(textposition="outside")
    return fig


def _plot_allocation_donut(allocation_df: pd.DataFrame) -> go.Figure:
    active_allocation = _display_allocation(allocation_df)
    fig = px.pie(
        active_allocation,
        names="ticker",
        values="weight",
        hole=0.48,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        template="plotly_white",
        title={"text": "Répartition du portefeuille recommandé", "x": 0.02},
        height=390,
        margin={"l": 10, "r": 10, "t": 60, "b": 20},
        showlegend=True,
        autosize=True,
    )
    return fig


def _plot_simulation_paths(paths_df: pd.DataFrame, target_value: float) -> go.Figure:
    fig = go.Figure()
    days = paths_df.index
    percentile_5 = paths_df.quantile(0.05, axis=1)
    median_path = paths_df.quantile(0.50, axis=1)
    percentile_95 = paths_df.quantile(0.95, axis=1)

    fig.add_trace(
        go.Scatter(
            x=days,
            y=percentile_95,
            mode="lines",
            line={"width": 0},
            name="Percentile 95",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=percentile_5,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(29, 78, 216, 0.16)",
            line={"width": 0},
            name="Bande 5%-95%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=median_path,
            mode="lines",
            line={"width": 3, "color": "#1d4ed8"},
            name="Médiane",
        )
    )

    columns_to_plot = list(paths_df.columns[: min(35, len(paths_df.columns))])
    for column in columns_to_plot:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=paths_df[column],
                mode="lines",
                line={"width": 1, "color": "rgba(15, 39, 71, 0.14)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_hline(
        y=target_value,
        line_dash="dash",
        line_color="#b91c1c",
        annotation_text="Objectif",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": "Trajectoires simulées du portefeuille", "x": 0.02},
        xaxis_title="Jour simulé",
        yaxis_title="Valeur du portefeuille (MAD)",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=480,
        autosize=True,
    )
    return fig


def _plot_final_value_distribution(paths_df: pd.DataFrame, target_value: float) -> go.Figure:
    final_values = paths_df.iloc[-1].astype(float)
    fig = px.histogram(
        final_values,
        nbins=45,
        labels={"value": "Valeur finale (MAD)", "count": "Scénarios"},
        color_discrete_sequence=["#1d4ed8"],
    )
    fig.add_vline(
        x=target_value,
        line_dash="dash",
        line_color="#b91c1c",
        annotation_text="Objectif",
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": "Distribution des valeurs finales simulées", "x": 0.02},
        showlegend=False,
        height=360,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        autosize=True,
    )
    return fig


def _plot_risk_contribution(returns_df: pd.DataFrame, allocation_df: pd.DataFrame) -> go.Figure | None:
    risk_df = _risk_contribution(returns_df, allocation_df)
    if risk_df.empty:
        return None

    risk_df = risk_df.sort_values("risk_contribution")
    fig = px.bar(
        risk_df,
        x="risk_contribution",
        y="ticker",
        orientation="h",
        text=risk_df["risk_contribution"].map(lambda value: f"{value:.1%}"),
        color="risk_contribution",
        color_continuous_scale=["#fef3c7", "#b45309"],
        labels={"risk_contribution": "Contribution au risque", "ticker": "Action"},
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": "Contribution estimée au risque par action", "x": 0.02},
        coloraxis_showscale=False,
        height=max(320, 32 * len(risk_df)),
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis_tickformat=".0%",
        autosize=True,
    )
    fig.update_traces(textposition="outside")
    return fig


def _display_simulation_summary(summary: dict) -> None:
    rows = [
        ("Valeur finale moyenne", _format_money(summary["mean_final_value"])),
        ("Valeur finale médiane", _format_money(summary["median_final_value"])),
        ("Percentile 5", _format_money(summary["percentile_5"])),
        ("Percentile 95", _format_money(summary["percentile_95"])),
        ("Perte maximale simulée", _format_money(summary["max_loss_simulated"])),
    ]
    if "probability_target_reached" in summary:
        rows.append(
            (
                "Probabilité d’atteindre l’objectif",
                _format_percent(summary["probability_target_reached"]),
            )
        )
    st.dataframe(
        pd.DataFrame(rows, columns=["Indicateur", "Valeur"]),
        hide_index=True,
        use_container_width=True,
    )


def _final_recommendation(probability: float | None, median_final_value: float, target_value: float) -> str:
    if probability is None:
        return "Aucune probabilité cible n’a été calculée."
    if probability >= 0.70 and median_final_value >= target_value:
        return (
            "La recommandation est favorable dans la simulation historique. "
            "Le scénario médian atteint l’objectif et la probabilité estimée est élevée."
        )
    if probability >= 0.40:
        return (
            "La recommandation est modérée. L’objectif peut être atteint, mais les résultats "
            "restent sensibles aux trajectoires de marché."
        )
    return (
        "La recommandation est prudente par rapport à l’objectif choisi. Un objectif plus bas, "
        "un horizon plus long ou un autre profil de risque peut être étudié."
    )


def _format_best_scored_stocks(scores_df: pd.DataFrame, top_n: int = 5) -> str:
    scores = scores_df.copy()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["ticker", "score"]).sort_values(
        "score",
        ascending=False,
    )
    if scores.empty:
        return "non disponible"

    top_scores = scores.head(top_n)
    return ", ".join(
        f"{row.ticker} ({row.score:.3f})"
        for row in top_scores.itertuples(index=False)
    )


def _format_recommended_allocation(allocation_df: pd.DataFrame, top_n: int = 6) -> str:
    allocation = _validated_allocation(allocation_df, require_score=True)
    allocation = allocation.sort_values("weight", ascending=False).reset_index(drop=True)
    significant_allocation = allocation[allocation["weight"] > ACTIVE_WEIGHT_THRESHOLD].copy()
    if significant_allocation.empty:
        significant_allocation = allocation.head(1).copy()

    top_allocation = significant_allocation.head(top_n)
    remaining_tickers = set(allocation["ticker"]) - set(top_allocation["ticker"])
    remaining_weight = float(
        allocation[allocation["ticker"].isin(remaining_tickers)]["weight"].sum()
    )

    allocation_parts = [
        f"{row.ticker} ({row.weight:.2%})"
        for row in top_allocation.itertuples(index=False)
    ]
    if remaining_weight > ACTIVE_WEIGHT_THRESHOLD:
        allocation_parts.append(f"autres titres ({remaining_weight:.2%})")

    return ", ".join(allocation_parts)


def _build_recommendation_limitations(
    allocation_df: pd.DataFrame,
    risk_metrics: dict,
    model_metrics: dict | None,
) -> list[str]:
    diagnostics = _allocation_diagnostics(allocation_df)
    limitations = [
        (
            "Les résultats reposent sur les prix et rendements historiques disponibles "
            f"({risk_metrics['observations']} observations utilisées pour le risque)."
        ),
        (
            "Le score logistique mesure une attractivité statistique relative ; il ne "
            "garantit pas la performance future des actions."
        ),
        (
            "La simulation Monte Carlo utilise un bootstrap historique : elle rééchantillonne "
            "les rendements passés sans supposer une loi normale, mais elle ne couvre pas tous "
            "les scénarios de marché possibles."
        ),
    ]

    if diagnostics["active_count"] < 5:
        limitations.append(
            f"Le portefeuille contient seulement {diagnostics['active_count']} actifs avec "
            "un poids significatif, ce qui limite la diversification."
        )
    if (
        diagnostics["max_weight"] >= CONCENTRATION_WEIGHT_THRESHOLD
        or diagnostics["effective_assets"] < CONCENTRATION_EFFECTIVE_ASSETS_THRESHOLD
    ):
        limitations.append(
            "L’allocation présente une concentration notable, avec un poids maximal de "
            f"{diagnostics['max_weight']:.2%} et {diagnostics['effective_assets']:.2f} "
            "actifs effectifs."
        )
    if model_metrics is None:
        limitations.append(
            "Les métriques du modèle ne sont pas disponibles dans cette session, ce qui "
            "réduit la capacité à juger la robustesse du scoring."
        )
    elif _model_is_weak(model_metrics):
        roc_auc = model_metrics.get("roc_auc")
        roc_auc_text = "N/A" if roc_auc is None else f"{roc_auc:.3f}"
        limitations.append(
            "Les métriques du modèle doivent être interprétées avec prudence "
            f"(F1={model_metrics.get('f1_score', 0):.3f}, ROC AUC={roc_auc_text})."
        )

    return limitations


def _render_final_interpretation(
    scores_df: pd.DataFrame,
    allocation_df: pd.DataFrame,
    risk_profile: str,
    risk_metrics: dict,
    summary: dict,
    target_value: float,
    model_metrics: dict | None,
) -> None:
    probability = summary.get("probability_target_reached")
    recommendation_text = _final_recommendation(
        probability,
        summary["median_final_value"],
        target_value,
    )
    limitations = _build_recommendation_limitations(
        allocation_df,
        risk_metrics,
        model_metrics,
    )
    limitations_text = "\n".join(f"- {item}" for item in limitations)

    st.markdown(
        f"""
**Synthèse générée automatiquement à partir des résultats affichés.**

Les actions les mieux scorées par le modèle sont : **{_format_best_scored_stocks(scores_df)}**.
Pour le profil de risque **{risk_profile}**, l’optimiseur recommande principalement :
**{_format_recommended_allocation(allocation_df)}**.

Le risque estimé du portefeuille correspond à une volatilité annualisée de
**{_format_percent(risk_metrics["annualized_volatility"])}** et une volatilité quotidienne de
**{_format_percent(risk_metrics["daily_volatility"])}**, calculées sur les rendements
historiques disponibles. La simulation Monte Carlo estime une probabilité de
**{_format_percent(probability)}** d’atteindre la valeur cible de **{_format_money(target_value)}**.
La valeur finale médiane simulée est **{_format_money(summary["median_final_value"])}**.

**Lecture de la recommandation.** {recommendation_text}

**Limites principales.**
{limitations_text}

**Disclaimer.** Cette application est un outil d’aide à la décision académique et ne constitue pas un conseil financier.
"""
    )


def _render_sidebar() -> tuple[float, int, str, float, str, int, bool]:
    st.sidebar.markdown("### Paramètres investisseur")
    st.sidebar.caption("Définissez le cadre de la recommandation.")

    budget = st.sidebar.number_input(
        "Budget d’investissement (MAD)",
        min_value=1.0,
        value=100000.0,
        step=1000.0,
    )
    horizon_days = st.sidebar.number_input(
        "Horizon d’investissement (jours)",
        min_value=1,
        max_value=1260,
        value=252,
        step=1,
    )
    risk_profile = st.sidebar.selectbox(
        "Profil de risque",
        ["conservative", "balanced", "dynamic"],
        index=1,
        help=(
            "Conservative limite davantage le poids par actif. Dynamic autorise une "
            "allocation plus concentrée."
        ),
    )
    target_value, target_label = _target_input(float(budget))
    n_simulations = st.sidebar.number_input(
        "Nombre de simulations Monte Carlo",
        min_value=100,
        max_value=20000,
        value=5000,
        step=500,
    )

    st.sidebar.markdown("---")
    run_button = st.sidebar.button(
        "Générer la recommandation",
        type="primary",
        use_container_width=True,
    )
    return (
        float(budget),
        int(horizon_days),
        risk_profile,
        float(target_value),
        target_label,
        int(n_simulations),
        run_button,
    )


def _render_ready_state(features_df: pd.DataFrame, returns_df: pd.DataFrame) -> None:
    feature_tickers = features_df["ticker"].dropna().astype(str).nunique()
    return_tickers = len([column for column in returns_df.columns if str(column) != "date"])
    latest_feature_date = features_df["date"].max() if "date" in features_df.columns else None

    st.markdown(
        """
        <div class="method-panel">
            <div class="section-title">Application prête</div>
            <p>
                Configurez les paramètres dans la barre latérale puis lancez la recommandation.
                Les résultats utiliseront les sorties actuelles du pipeline : features,
                rendements historiques, modèle logistique et scaler.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(3)
    with columns[0]:
        _metric_card("Actions avec features", _format_number(feature_tickers), "Univers scoré")
    with columns[2]:
        value = latest_feature_date.strftime("%Y-%m-%d") if pd.notna(latest_feature_date) else "N/A"
        _metric_card("Dernière date features", value, "Fraîcheur des signaux")


def _render_kpis(
    scores_df: pd.DataFrame,
    risk_profile: str,
    portfolio_score: float,
    risk_metrics: dict,
    probability: float | None,
) -> None:
    columns = st.columns(5)
    with columns[0]:
        _metric_card("Actions scorées", _format_number(scores_df["ticker"].nunique()), "Univers analysé")
    with columns[1]:
        _metric_card("Profil sélectionné", risk_profile, "Contraintes de risque")
    with columns[2]:
        _metric_card("Score attendu", f"{portfolio_score:.3f}", "Score pondéré du portefeuille")
    with columns[3]:
        _metric_card("Volatilité annualisée", _format_percent(risk_metrics["annualized_volatility"]), "Risque historique")
    with columns[4]:
        _metric_card("Probabilité cible", _format_percent(probability), "Bootstrap historique")


def _format_scores_table(scores_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    table = scores_df.copy()
    table["score"] = pd.to_numeric(table["score"], errors="coerce")
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    table["date"] = pd.to_datetime(table["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    def highlight_top_five(row: pd.Series) -> list[str]:
        style = "background-color: #eff6ff; font-weight: 700; color: #0f2747;"
        return [style if row.name < 5 else "" for _ in row]

    return (
        table.style.format({"score": "{:.3f}"}, na_rep="N/A")
        .apply(highlight_top_five, axis=1)
    )


def _format_allocation_table(allocation_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    table = allocation_df.copy()
    table["weight"] = pd.to_numeric(table["weight"], errors="coerce")
    table["score"] = pd.to_numeric(table["score"], errors="coerce")
    table = table.sort_values("weight", ascending=False).reset_index(drop=True)
    return table.style.format({"weight": "{:.2%}", "score": "{:.3f}"}, na_rep="N/A")


def _format_investment_table(investment_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    table = investment_df.copy()
    table["weight"] = pd.to_numeric(table["weight"], errors="coerce")
    table["score"] = pd.to_numeric(table["score"], errors="coerce")
    table["amount"] = pd.to_numeric(table["amount"], errors="coerce")
    table["amount_rounded"] = pd.to_numeric(table["amount_rounded"], errors="coerce")
    table = table.sort_values("weight", ascending=False).reset_index(drop=True)
    return table.style.format(
        {
            "weight": "{:.2%}",
            "score": "{:.3f}",
            "amount": _format_money,
            "amount_rounded": _format_money,
        },
        na_rep="N/A",
    )


def main() -> None:
    st.set_page_config(
        page_title="Optimisation intelligente d'un portefeuille",
        page_icon="📈",
        layout="wide",
    )
    _inject_css()
    _render_header()

    if not _check_required_files():
        st.stop()

    (
        budget,
        horizon_days,
        risk_profile,
        target_value,
        target_label,
        n_simulations,
        run_button,
    ) = _render_sidebar()

    try:
        features_df = _load_features(FEATURES_PATH)
        returns_df = _load_returns(RETURNS_PATH)
        model = _load_pickle(MODEL_PATH)
        scaler = _load_pickle(SCALER_PATH)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    _warn_if_latest_scores_file_is_stale()

    if not run_button:
        _render_ready_state(features_df, returns_df)
        return

    with st.spinner("Calcul de la recommandation en cours..."):
        try:
            scores_df = score_latest_stocks(features_df, model, scaler)
            _warn_if_three_ticker_scores(scores_df, "latest_scores.csv")
            scores_df = _align_scores_with_returns(scores_df, returns_df)
            allocation_df = optimize_portfolio(scores_df, returns_df, risk_profile=risk_profile)
            investment_df = compute_investment_amounts(allocation_df, budget)
            risk_metrics = _calculate_portfolio_risk(returns_df, allocation_df)
            paths_df, summary = historical_bootstrap_monte_carlo(
                returns_df=returns_df,
                allocation_df=allocation_df,
                initial_value=budget,
                horizon_days=horizon_days,
                n_simulations=n_simulations,
                target_value=target_value,
                random_state=42,
            )
            model_metrics = _compute_model_metrics(model, scaler)
        except Exception as exc:
            st.error(
                "Impossible de générer la recommandation : "
                f"{_helpful_recommendation_error(exc, risk_profile)}"
            )
            st.stop()

    probability = summary.get("probability_target_reached")
    portfolio_score = _portfolio_expected_score(allocation_df)
    allocation_diag = _allocation_diagnostics(allocation_df)

    _render_kpis(scores_df, risk_profile, portfolio_score, risk_metrics, probability)
    _render_warning_panels(allocation_df, model_metrics)

    overview_tab, scores_tab, allocation_tab, simulation_tab, diagnostics_tab = st.tabs(
        [
            "Vue d’ensemble",
            "Scores des actions",
            "Allocation optimale",
            "Simulation Monte Carlo",
            "Diagnostic & limites",
        ]
    )

    with overview_tab:
        st.markdown('<div class="section-title">Synthèse de la recommandation</div>', unsafe_allow_html=True)
        left_column, right_column = st.columns([1.05, 1])
        with left_column:
            st.markdown(
                """
                <div class="info-panel">
                    <strong>Lecture rapide.</strong> Le score mesure l’attractivité relative
                    estimée par le modèle logistique à partir des facteurs financiers historiques.
                    L’allocation transforme ces scores en poids de portefeuille sous contraintes
                    de risque, sans vente à découvert.
                </div>
                """,
                unsafe_allow_html=True,
            )
            summary_rows = pd.DataFrame(
                [
                    ("Budget", _format_money(budget)),
                    ("Objectif", target_label),
                    ("Valeur cible", _format_money(target_value)),
                    ("Score attendu du portefeuille", f"{portfolio_score:.3f}"),
                    ("Actifs significatifs", allocation_diag["active_count"]),
                    ("Nombre de simulations", _format_number(n_simulations)),
                ],
                columns=["Paramètre", "Valeur"],
            )
            st.dataframe(summary_rows, hide_index=True, use_container_width=True)

        with right_column:
            st.plotly_chart(_plot_allocation_donut(allocation_df), use_container_width=True)
            st.caption(
                "Ce graphique montre la répartition synthétique des poids optimisés. "
                "Les très petites positions peuvent être regroupées sous Autres pour préserver la lisibilité."
            )

        st.markdown('<div class="section-title">Recommandation finale</div>', unsafe_allow_html=True)
        _render_final_interpretation(
            scores_df=scores_df,
            allocation_df=allocation_df,
            risk_profile=risk_profile,
            risk_metrics=risk_metrics,
            summary=summary,
            target_value=target_value,
            model_metrics=model_metrics,
        )

    with scores_tab:
        st.markdown('<div class="section-title">Scores des actions</div>', unsafe_allow_html=True)
        st.caption(
            "Un score élevé indique une attractivité relative plus forte selon le modèle "
            "logistique entraîné sur les facteurs financiers. Il ne constitue pas une garantie "
            "de performance future."
        )
        st.plotly_chart(_plot_scores(scores_df), use_container_width=True)
        st.caption(
            "Les barres comparent les actions selon leur score d’attractivité estimé. "
            "Le tableau ci-dessous est trié du score le plus élevé au plus faible, avec les cinq premiers titres surlignés."
        )
        st.dataframe(
            _format_scores_table(scores_df),
            hide_index=True,
            use_container_width=True,
        )

    with allocation_tab:
        st.markdown('<div class="section-title">Allocation optimale</div>', unsafe_allow_html=True)
        st.caption(
            "L’allocation recommandée respecte les contraintes du profil de risque choisi. "
            "Les graphiques regroupent les poids quasi nuls sous Autres lorsque nécessaire."
        )
        st.plotly_chart(_plot_allocation_bar(allocation_df), use_container_width=True)
        st.caption(
            "Le graphique présente les principaux poids retenus par l’optimiseur. "
            "Le tableau conserve l’allocation complète triée par poids décroissant."
        )

        left_column, right_column = st.columns([1.2, 1])
        with left_column:
            st.subheader("Poids recommandés")
            st.dataframe(
                _format_allocation_table(allocation_df),
                hide_index=True,
                use_container_width=True,
            )
        with right_column:
            st.subheader("Montants à investir")
            st.dataframe(
                _format_investment_table(investment_df),
                hide_index=True,
                use_container_width=True,
            )

        risk_figure = _plot_risk_contribution(returns_df, allocation_df)
        if risk_figure is not None:
            st.subheader("Contribution au risque")
            st.plotly_chart(risk_figure, use_container_width=True)
            st.caption(
                "Cette vue indique quelles actions contribuent le plus au risque historique "
                "du portefeuille optimisé."
            )

    with simulation_tab:
        st.markdown('<div class="section-title">Simulation Monte Carlo par bootstrap historique</div>', unsafe_allow_html=True)
        st.caption(
            "Les trajectoires sont simulées en rééchantillonnant les rendements historiques "
            "du portefeuille optimisé. La bande représente les percentiles 5%-95%."
        )
        st.plotly_chart(
            _plot_simulation_paths(paths_df, target_value),
            use_container_width=True,
        )
        st.caption(
            "La courbe bleue représente la trajectoire médiane ; la zone bleutée couvre "
            "les scénarios entre les percentiles 5 et 95."
        )
        st.plotly_chart(
            _plot_final_value_distribution(paths_df, target_value),
            use_container_width=True,
        )
        st.caption(
            "L’histogramme résume la distribution des valeurs finales simulées et situe "
            "la valeur cible par rapport aux scénarios générés."
        )
        _display_simulation_summary(summary)

    with diagnostics_tab:
        st.markdown('<div class="section-title">Diagnostic & limites</div>', unsafe_allow_html=True)
        diagnostic_columns = st.columns(4)
        with diagnostic_columns[0]:
            _metric_card("Volatilité quotidienne", _format_percent(risk_metrics["daily_volatility"]), "Historique du portefeuille")
        with diagnostic_columns[1]:
            _metric_card("Variance", f"{risk_metrics['variance']:.6f}", "Risque quotidien")
        with diagnostic_columns[2]:
            _metric_card("Observations risque", _format_number(risk_metrics["observations"]), "Rendements utilisés")
        with diagnostic_columns[3]:
            _metric_card("Actifs effectifs", f"{allocation_diag['effective_assets']:.2f}", "Diversification")

        st.subheader("Métriques du modèle")
        if model_metrics:
            model_rows = [
                ("Accuracy", f"{model_metrics['accuracy']:.4f}"),
                ("Precision", f"{model_metrics['precision']:.4f}"),
                ("Recall", f"{model_metrics['recall']:.4f}"),
                ("F1 score", f"{model_metrics['f1_score']:.4f}"),
                (
                    "ROC AUC",
                    "N/A" if model_metrics["roc_auc"] is None else f"{model_metrics['roc_auc']:.4f}",
                ),
                ("Confusion matrix", str(model_metrics["confusion_matrix"])),
            ]
            st.dataframe(
                pd.DataFrame(model_rows, columns=["Métrique", "Valeur"]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning(
                "Les métriques du modèle ne sont pas disponibles dans cette session. "
                "Le pipeline peut être relancé pour les recalculer."
            )

        st.markdown(
            f"""
            <div class="method-panel">
                <strong>Limites méthodologiques.</strong>
                Les résultats dépendent des données historiques disponibles, de la qualité
                du modèle de scoring et des contraintes d’optimisation. Le bootstrap historique
                ne suppose pas une distribution normale, mais il ne garantit pas que les futurs
                marchés reproduiront les scénarios passés.
                <br><br>
                Période utilisée pour le risque :
                <code>{risk_metrics['start_date']:%Y-%m-%d}</code> à
                <code>{risk_metrics['end_date']:%Y-%m-%d}</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
