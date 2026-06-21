"""Streamlit preview app for raw BVC prices after Phase 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PRICES_PATH = PROJECT_ROOT / "data" / "raw" / "raw_prices.csv"
COLLECTION_REPORT_PATH = PROJECT_ROOT / "data" / "outputs" / "collection_report.csv"
EXCLUDED_TICKERS_REPORT_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "excluded_tickers_report.csv"
)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column, allowing case-insensitive fallback."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_to_original = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        match = lower_to_original.get(candidate.lower())
        if match is not None:
            return match

    return None


@st.cache_data
def _load_raw_prices(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def _load_raw_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _is_valid_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _show_missing_file_message() -> None:
    st.error(f"`{_relative_path(RAW_PRICES_PATH)}` is missing or empty.")
    st.info("Run Phase 1 first to collect raw BVC prices, then reload this page.")


def _date_series(df: pd.DataFrame, date_column: str | None) -> pd.Series:
    if date_column is None:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[date_column], errors="coerce", dayfirst=True)


def _display_data_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview")

    ticker_column = _find_column(df, ["ticker", "Ticker"])
    total_companies = (
        df[ticker_column].dropna().astype(str).nunique() if ticker_column is not None else 0
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric("Rows", f"{df.shape[0]:,}")
    metric_columns[1].metric("Columns", f"{df.shape[1]:,}")
    metric_columns[2].metric("Companies", f"{total_companies:,}")

    st.write("Columns:")
    st.code(", ".join(map(str, df.columns)))

    date_column = _find_column(df, ["date", "Date"])
    if date_column is not None:
        dates = pd.to_datetime(df[date_column], errors="coerce", dayfirst=True).dropna()
        if dates.empty:
            st.warning("No valid dates could be parsed from the raw data.")
        else:
            st.write(f"Date range: `{dates.min().date()}` to `{dates.max().date()}`")
    else:
        st.warning("No date column was found.")

    if ticker_column is not None:
        st.write("Rows per company:")
        st.dataframe(
            df.groupby(ticker_column, dropna=False)
            .size()
            .rename("rows")
            .sort_values(ascending=False),
            use_container_width=True,
        )
    else:
        st.warning("No ticker column was found.")

    st.write("Missing values summary:")
    st.dataframe(
        df.isna().sum().rename("missing_values"),
        use_container_width=True,
    )


def _filter_data(df: pd.DataFrame) -> pd.DataFrame:
    ticker_column = _find_column(df, ["ticker", "Ticker"])
    date_column = _find_column(df, ["date", "Date"])
    filtered = df.copy()

    if ticker_column is not None:
        tickers = sorted(filtered[ticker_column].dropna().astype(str).unique())
        default_tickers = tickers[: min(10, len(tickers))]
        selected_tickers = st.sidebar.multiselect(
            "Ticker filter",
            options=tickers,
            default=default_tickers,
        )
        if not selected_tickers:
            st.warning("Select at least one ticker to preview rows and chart prices.")
            return filtered.iloc[0:0]
        filtered = filtered[filtered[ticker_column].astype(str).isin(selected_tickers)].copy()

    parsed_dates = _date_series(filtered, date_column)
    valid_dates = parsed_dates.dropna()
    if date_column is not None and not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        selected_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
            filtered = filtered[
                parsed_dates.between(
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date),
                    inclusive="both",
                )
            ].copy()
        else:
            st.sidebar.info("Select a start and end date to apply the date filter.")

    return filtered


def _display_price_chart(df: pd.DataFrame) -> None:
    date_column = _find_column(df, ["date", "Date"])
    ticker_column = _find_column(df, ["ticker", "Ticker"])
    price_column = _find_column(df, ["close", "Value"])

    if date_column is None or ticker_column is None or price_column is None:
        st.info("Price chart requires date, ticker, and close or Value columns.")
        return

    chart_df = df[[date_column, ticker_column, price_column]].copy()
    chart_df[date_column] = pd.to_datetime(
        chart_df[date_column],
        errors="coerce",
        dayfirst=True,
    )
    chart_df[price_column] = pd.to_numeric(chart_df[price_column], errors="coerce")
    chart_df = chart_df.dropna(subset=[date_column, ticker_column, price_column])
    chart_df = chart_df.sort_values([ticker_column, date_column])

    if chart_df.empty:
        st.warning("No valid rows are available for the price chart.")
        return

    st.subheader("Price Over Time")
    fig = px.line(
        chart_df,
        x=date_column,
        y=price_column,
        color=ticker_column,
        markers=False,
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Ticker",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_volume_chart(df: pd.DataFrame) -> None:
    date_column = _find_column(df, ["date", "Date"])
    ticker_column = _find_column(df, ["ticker", "Ticker"])
    volume_column = _find_column(df, ["volume", "Volume"])

    if date_column is None or ticker_column is None or volume_column is None:
        st.info("Volume chart requires date, ticker, and volume or Volume columns.")
        return

    chart_df = df[[date_column, ticker_column, volume_column]].copy()
    chart_df[date_column] = pd.to_datetime(
        chart_df[date_column],
        errors="coerce",
        dayfirst=True,
    )
    chart_df[volume_column] = pd.to_numeric(chart_df[volume_column], errors="coerce")
    chart_df = chart_df.dropna(subset=[date_column, ticker_column, volume_column])
    chart_df = chart_df.sort_values([ticker_column, date_column])

    if chart_df.empty:
        st.warning("No valid rows are available for the volume chart.")
        return

    st.subheader("Volume Over Time")
    fig = px.line(
        chart_df,
        x=date_column,
        y=volume_column,
        color=ticker_column,
        markers=False,
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
        legend_title="Ticker",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_optional_report(path: Path, title: str) -> None:
    if not _is_valid_file(path):
        return

    st.subheader(title)
    try:
        report = _load_csv(path)
    except Exception as exc:
        st.warning(f"Could not load `{_relative_path(path)}`: {exc}")
        return

    st.dataframe(report, use_container_width=True)


def _download_button_for_file(path: Path, label: str) -> None:
    if not _is_valid_file(path):
        return

    st.download_button(
        label,
        data=_load_raw_bytes(path),
        file_name=path.name,
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Raw BVC Data Preview", layout="wide")
    st.title("Raw BVC Data Preview")

    if not _is_valid_file(RAW_PRICES_PATH):
        _show_missing_file_message()
        st.stop()

    try:
        raw_prices = _load_raw_prices(RAW_PRICES_PATH)
    except Exception as exc:
        st.error(f"Could not load `{_relative_path(RAW_PRICES_PATH)}`: {exc}")
        st.info("Run Phase 1 first to regenerate the raw prices file.")
        st.stop()

    if raw_prices.empty:
        _show_missing_file_message()
        st.stop()

    st.sidebar.header("Filters")
    filtered_prices = _filter_data(raw_prices)

    _display_data_overview(raw_prices)

    st.subheader("First Rows")
    st.write(f"Filtered shape: `{filtered_prices.shape}`")
    st.dataframe(filtered_prices.head(20), use_container_width=True)

    _display_price_chart(filtered_prices)
    _display_volume_chart(filtered_prices)

    _display_optional_report(COLLECTION_REPORT_PATH, "Collection Report")
    _display_optional_report(EXCLUDED_TICKERS_REPORT_PATH, "Excluded Tickers Report")

    st.subheader("Downloads")
    _download_button_for_file(RAW_PRICES_PATH, "Download raw_prices.csv")
    _download_button_for_file(COLLECTION_REPORT_PATH, "Download collection_report.csv")
    _download_button_for_file(
        EXCLUDED_TICKERS_REPORT_PATH,
        "Download excluded_tickers_report.csv",
    )


if __name__ == "__main__":
    main()
