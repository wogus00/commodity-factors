"""Streamlit showcase consuming saved analysis; widget changes never fit models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from commodity_factors.analysis import MODEL_LABELS, TABLES, load_bundle, summarize_predictions
from commodity_factors.daily import daily_snapshot
from commodity_factors.schwartz_smith import futures_log_price


ROOT = Path(__file__).resolve().parent
COLORS = {"Observed": "#2fc6ad", "Random walk": "#e4b05c", "One factor": "#b695f9",
          "Two factors": "#5b8cff", "Baseline": "#aeb9cf", "Hypothetical": "#5b8cff"}


@st.cache_data(show_spinner=False)
def read_analysis(directory, revision):
    return load_bundle(directory)


def available_datasets():
    paths = sorted((ROOT / "results").glob("*/metadata.json"))
    demo = ROOT / "data/synthetic/demo/metadata.json"
    if demo.exists():
        paths.append(demo)
    choices = {}
    for path in paths:
        try:
            metadata = json.loads(path.read_text())
            label = metadata["dataset_label"]
            if label in choices:
                label = f"{label} ({path.parent.name})"
            choices[label] = path.parent
        except (ValueError, KeyError):
            continue
    return choices


def chart(fig, key):
    fig.update_layout(
        margin=dict(l=12, r=12, t=24, b=12), height=370,
        legend=dict(orientation="h", y=1.13, x=0),
        font=dict(family="Arial, sans-serif", color="#aeb9cf"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#2a2e39", zerolinecolor="#3d4558")
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def add_periods(fig, bundle):
    meta = bundle.metadata
    end = bundle.panel["date_"].max()
    fig.add_vrect(x0=meta["validation_start"], x1=meta["test_start"],
                  fillcolor="#d1ddef", opacity=0.25, line_width=0, layer="below")
    fig.add_vrect(x0=meta["test_start"], x1=end,
                  fillcolor="#bde4dd", opacity=0.25, line_width=0, layer="below")


def model_lines(frame, x, y, observed_column):
    observed = frame.drop_duplicates(x).sort_values(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=observed[x], y=observed[observed_column],
                            name="Observed", line=dict(color=COLORS["Observed"], width=2.5)))
    for model, group in frame.groupby("model", sort=False):
        group = group.sort_values(x)
        label = MODEL_LABELS[model]
        fig.add_trace(go.Scatter(x=group[x], y=group[y], name=label,
                                line=dict(color=COLORS[label], width=2, dash="dash")))
    return fig


def overview(bundle, rank):
    panel = bundle.panel
    st.subheader("The market through time")
    observed = panel[panel.maturity_rank == rank]
    fig = px.line(observed, x="date_", y="settlement",
                  labels={"date_": "Date", "settlement": "USD / barrel"})
    fig.update_traces(line_color=COLORS["Observed"])
    add_periods(fig, bundle)
    chart(fig, "market-history")
    st.caption(f"Maturity rank {rank}. Blue shading: validation. Green shading: test. "
               "Rank histories change contract at rolls; they are not spot-price series.")

    st.subheader("Explore a futures curve")
    dates = sorted(panel.date_.unique())
    date = st.select_slider("Curve date", options=dates, value=dates[-1],
                            format_func=lambda value: pd.Timestamp(value).strftime("%b %d, %Y"))
    curve = panel[panel.date_ == date].copy()
    curve["days_to_expiry"] = curve.tau * 365.25
    left, right = st.columns([3, 2])
    with left:
        fig = px.line(curve, x="days_to_expiry", y="settlement", markers=True,
                      labels={"days_to_expiry": "Days to last trading date", "settlement": "USD / barrel"})
        fig.update_traces(line_color=COLORS["Two factors"])
        chart(fig, "market-curve")
    with right:
        front, back = curve.iloc[0].settlement, curve.iloc[-1].settlement
        st.metric("Front contract", f"${front:,.2f}")
        st.metric("Last / front spread", f"${back - front:+,.2f}")
        st.caption("An upward curve is contango; a downward curve is backwardation.")
        columns = ["maturity_rank", "futcode", "lasttrddate", "settlement"]
        if "dsmnem" in curve:
            columns.insert(2, "dsmnem")
        st.dataframe(curve[columns], hide_index=True, width="stretch")


def forecasts(bundle, predictions, rank, models, horizon):
    selected = predictions[(predictions.maturity_rank == rank) & predictions.model.isin(models)].copy()
    scale = st.radio("Price scale", ["USD / barrel", "Log price"], horizontal=True)
    for source, destination in [("observed_log", "Observed"), ("predicted_log", "Prediction")]:
        selected[destination] = np.exp(selected[source]) if scale == "USD / barrel" else selected[source]
    st.subheader(f"{horizon}-day-ahead forecasts · maturity rank {rank}")
    fig = model_lines(selected, "date_", "Prediction", "Observed")
    fig.update_layout(xaxis_title="Target date", yaxis_title=scale)
    chart(fig, "forecast-history")
    st.caption("Each prediction uses the state available at its origin, then advances without "
               "intermediate price observations. Price-level lines show exponentiated log forecasts.")

    st.subheader("Forecast curve on a selected day")
    dates = sorted(predictions.date_.unique())
    date = st.select_slider("Target date", options=dates, value=dates[-1],
                            format_func=lambda value: pd.Timestamp(value).strftime("%b %d, %Y"))
    curve = predictions[(predictions.date_ == date) & predictions.model.isin(models)].copy()
    curve["days_to_expiry"] = curve.tau * 365.25
    curve["Observed"] = np.exp(curve.observed_log) if scale == "USD / barrel" else curve.observed_log
    curve["Prediction"] = np.exp(curve.predicted_log) if scale == "USD / barrel" else curve.predicted_log
    fig = model_lines(curve, "days_to_expiry", "Prediction", "Observed")
    fig.update_layout(xaxis_title="Days to last trading date", yaxis_title=scale)
    chart(fig, "forecast-curve")
    origin = pd.Timestamp(curve.origin_date.iloc[0]).strftime("%b %d, %Y")
    st.caption(f"Forecast origin: {origin}. All models target the same ranked contracts on this date.")

    with st.expander("Same-day curve reconstruction"):
        st.caption("These fitted curves use the selected day's prices. They measure fit and are not forecasts.")
        fitted = bundle.reconstructions[bundle.reconstructions.date_ == date].copy()
        fitted = fitted[fitted.model.isin(models)]
        if fitted.empty:
            st.info("Select a factor model to display reconstruction.")
        else:
            fitted["days_to_expiry"] = fitted.tau * 365.25
            fitted["Observed"] = np.exp(fitted.observed_log)
            fitted["Fitted"] = np.exp(fitted.fitted_log)
            fig = model_lines(fitted, "days_to_expiry", "Fitted", "Observed")
            fig.update_layout(xaxis_title="Days to last trading date", yaxis_title="USD / barrel")
            chart(fig, "reconstruction")


def performance(bundle, predictions, models, split, horizon, date_window):
    units = st.radio("Error units", ["Log price", "USD / barrel"], horizontal=True)
    suffix = "log" if units == "Log price" else "price"
    metrics = summarize_predictions(predictions)
    overall = metrics[metrics.maturity_rank == 0].set_index("model")
    rw_rmse = overall.loc["random_walk", f"rmse_{suffix}"]
    summary = overall.loc[models, [f"rmse_{suffix}", f"mae_{suffix}", "n_dates", "n"]].copy()
    summary["Change vs random walk (%)"] = np.where(
        rw_rmse > 0, (1 - summary[f"rmse_{suffix}"] / rw_rmse) * 100, np.nan)
    summary.index = summary.index.map(MODEL_LABELS)
    summary = summary.rename(columns={f"rmse_{suffix}": "RMSE", f"mae_{suffix}": "MAE",
                                       "n_dates": "Dates", "n": "Observations"})
    st.subheader("Forecast accuracy")
    st.dataframe(summary, width="stretch")
    st.caption(f"{split.title()} · {horizon} trading days ahead · {units.lower()}. "
               "Positive change means a lower RMSE than random walk. Every model uses the same sample.")

    st.subheader("Where errors occur along the curve")
    ranked = metrics[(metrics.maturity_rank > 0) & metrics.model.isin(models)].copy()
    ranked["Model"] = ranked.model.map(MODEL_LABELS)
    fig = px.bar(ranked, x="maturity_rank", y=f"rmse_{suffix}", color="Model", barmode="group",
                 color_discrete_map=COLORS,
                 labels={"maturity_rank": "Maturity rank", f"rmse_{suffix}": f"RMSE · {units}"})
    chart(fig, "rank-errors")

    st.subheader("Forecast horizon comparison")
    all_horizons = bundle.predictions[
        (bundle.predictions.split == split)
        & (bundle.predictions.date_ >= pd.Timestamp(date_window[0]))
        & (bundle.predictions.date_ <= pd.Timestamp(date_window[1]))]
    by_horizon = summarize_predictions(all_horizons)
    by_horizon = by_horizon[(by_horizon.maturity_rank == 0) & by_horizon.model.isin(models)].copy()
    by_horizon["Model"] = by_horizon.model.map(MODEL_LABELS)
    fig = px.line(by_horizon, x="horizon", y=f"rmse_{suffix}", color="Model", markers=True,
                  color_discrete_map=COLORS,
                  labels={"horizon": "Forecast horizon · trading days", f"rmse_{suffix}": f"RMSE · {units}"})
    chart(fig, "horizon-errors")
    st.caption("Within each horizon, all models share the same sample. Longer horizons can have "
               "fewer early validation dates because forecasts must originate after calibration.")
    st.download_button("Download forecasts (CSV)", predictions.to_csv(index=False).encode(),
                       file_name=f"{bundle.metadata['dataset_id']}_{split}_{horizon}d.csv", mime="text/csv")


def exploration(bundle, split):
    states = bundle.states
    st.subheader("Estimated latent factors")
    left, right = st.columns(2)
    for column, label, container in [("chi", "Short-term deviation · χ", left),
                                      ("xi", "Long-term equilibrium · ξ", right)]:
        with container:
            st.caption(label)
            fig = px.line(states, x="date_", y=column, labels={"date_": "Date", column: "Log price"})
            fig.update_traces(line_color=COLORS["Two factors"])
            add_periods(fig, bundle)
            chart(fig, f"factor-{column}")
    st.caption("Filtered states are model-dependent estimates using information available through each date.")

    st.subheader("Hypothetical factor shocks")
    dates = sorted(bundle.panel.date_.unique())
    date = st.select_slider("Scenario date", options=dates, value=dates[-1],
                            format_func=lambda value: pd.Timestamp(value).strftime("%b %d, %Y"))
    left, right = st.columns(2)
    short = left.slider("Short-term shock · log points", -0.30, 0.30, 0.0, 0.01)
    long = right.slider("Long-term shock · log points", -0.30, 0.30, 0.0, 0.01)
    state = states.loc[states.date_ == date, ["chi", "xi"]].iloc[0].to_numpy(dtype=float)
    maturity = bundle.panel.loc[bundle.panel.date_ == date, "tau"].to_numpy(dtype=float)
    params = bundle.metadata["parameters"]["two_factor"]
    scenario = pd.DataFrame({"Days to last trading date": maturity * 365.25,
                             "Baseline": np.exp(futures_log_price(state, maturity, params)),
                             "Hypothetical": np.exp(futures_log_price(state + [short, long], maturity, params))})
    fig = px.line(scenario, x="Days to last trading date", y=["Baseline", "Hypothetical"],
                  color_discrete_map=COLORS, labels={"value": "USD / barrel", "variable": "Curve"})
    chart(fig, "factor-scenario")
    st.caption("A hypothetical change to the filtered state, with fitted parameters held fixed. "
               "Short-term effects decay with maturity; long-term shifts affect the whole curve. This is not a forecast.")

    with st.expander("Fitted parameters and calibration diagnostics"):
        params = bundle.metadata["parameters"]
        st.dataframe(pd.DataFrame({MODEL_LABELS[k]: pd.Series(v) for k, v in params.items()}), width="stretch")
        diagnostics = bundle.metadata["calibration"]
        for model in ("one_factor", "two_factor"):
            fit = diagnostics.get(model, {})
            st.write(f"**{MODEL_LABELS[model]}:** {fit.get('message', 'Fixed parameters')}")
        st.caption("Drift and mean reversion are annualized; volatility is per square root of a year. "
                   "Measurement noise is in log-price units. The two-factor calibration fixes λχ = 0.")
        holdout = states[states.split == split]
        st.dataframe(pd.DataFrame({"Model": ["One factor", "Two factors"],
                                   "One-step predictive log likelihood": [
                                       holdout.one_factor_predictive_ll.sum(), holdout.two_factor_predictive_ll.sum()]}),
                     hide_index=True, width="stretch")
        st.caption(f"Likelihood covers the full {split} period and one-step predictions, regardless of the selected horizon.")


def daily_forecasts(bundle, rank, models):
    dates = sorted(bundle.panel.loc[bundle.panel.date_ >= pd.Timestamp(bundle.metadata["train_end"]), "date_"].unique())
    chosen = st.date_input("Observed date", value=pd.Timestamp(dates[-1]).date(),
                           min_value=pd.Timestamp(dates[0]).date(), max_value=pd.Timestamp(dates[-1]).date(),
                           key=f"daily-observed-{bundle.metadata['dataset_id']}")
    available = [pd.Timestamp(date) for date in dates if pd.Timestamp(date).date() <= chosen]
    as_of = available[-1]
    snapshot = daily_snapshot(bundle, as_of, rank)
    target = st.date_input("Next forecast date", value=pd.Timestamp(snapshot["target_date"]).date(),
                           min_value=(as_of + pd.Timedelta(days=1)).date(),
                           key=f"daily-target-{bundle.metadata['dataset_id']}-{as_of.date()}")
    if pd.Timestamp(target) <= as_of:
        st.info("Choose a forecast date after the observed date.")
        return
    if str(target) != snapshot["target_date"]:
        snapshot = daily_snapshot(bundle, as_of, rank, target)
    if as_of.date() != chosen:
        st.caption(f"No observation on {chosen}; showing the latest available settlement, {as_of.date()}.")
    st.caption("Synthetic replay" if bundle.metadata["dataset_id"] == "synthetic" else
               f"Saved data through {bundle.panel.date_.max().date()} · Automatic feed not connected")
    left, middle, right = st.columns(3)
    left.metric(f"Observed · {snapshot['as_of']}", f"${snapshot['observed_price']:,.2f}")
    middle.markdown(f"**Forecast review · {snapshot['as_of']}**")
    middle.caption(f"{'Recorded' if snapshot['prior_recorded_at'] else 'Reconstructed'} forecast · "
                   f"Data through {snapshot['prior_origin']}" if snapshot["prior_origin"] else "No prior forecast after calibration")
    if snapshot["prior_recorded_at"]:
        middle.caption(f"Recorded at {snapshot['prior_recorded_at']}")
    right.markdown(f"**Next forecast · {snapshot['target_date']}**")
    right.caption(snapshot["calendar_basis"])
    st.caption(f"{snapshot['contract_label']} · Last trading date {snapshot['expiry']}")
    rows = pd.DataFrame(snapshot["models"])
    display = rows[rows.model.isin(models)][["model", "prior_price", "error_price", "error_percent", "next_price"]].copy()
    display["model"] = display.model.map(MODEL_LABELS)
    display.columns = ["Model", "Prior forecast · USD", "Error · USD", "Error · %", "Next forecast · USD"]
    st.dataframe(display, hide_index=True, width="stretch")
    columns = st.columns(3)
    for column, (name, key) in zip(columns, [("One-factor state x", "one_factor_state"), ("Short-term χ", "chi"), ("Long-term ξ", "xi")]):
        column.metric(name, f"{snapshot['state'][key]:.5f}")
    st.caption("Error = observed − prior forecast; % divides by the observed price. Factors update with "
               "the observed settlement; fitted parameters stay fixed. Price forecasts are conditional medians.")
    if snapshot["expired_before_target"]:
        st.info("The selected contract expires before the target. Select a later maturity rank.")
    quote = bundle.panel[(bundle.panel.date_ <= as_of) & (bundle.panel.futcode.astype(str) == str(
        bundle.panel[(bundle.panel.date_ == as_of) & (bundle.panel.maturity_rank == rank)].iloc[0].futcode))].tail(60)
    fig = go.Figure([go.Scatter(x=quote.date_, y=quote.settlement, name="Observed", line=dict(color=COLORS["Observed"]))])
    for row in snapshot["models"]:
        if row["model"] not in models:
            continue
        label = MODEL_LABELS[row["model"]]
        if row["prior_price"] is not None:
            fig.add_trace(go.Scatter(x=[as_of], y=[row["prior_price"]], name=f"{label} · Prior forecast", mode="markers",
                                     marker=dict(color=COLORS[label], symbol="circle-open", size=10)))
        if row["next_price"] is not None:
            fig.add_trace(go.Scatter(x=[as_of, snapshot["target_date"]], y=[snapshot["observed_price"], row["next_price"]],
                                     name=f"{label} · Next forecast", mode="lines+markers",
                                     line=dict(color=COLORS[label], dash="dot"), marker=dict(symbol="diamond", size=7)))
    fig.update_layout(xaxis_title="Settlement date", yaxis_title=bundle.metadata["price_unit"])
    chart(fig, "daily-forecast")
    st.caption("The contract stays fixed across the observation and forecast dates. Circles show prior forecasts; diamonds show next forecasts.")
    with st.expander("Current fitted parameters"):
        st.caption(f"Calibrated through {bundle.metadata['train_end']}")
        st.dataframe(pd.DataFrame({MODEL_LABELS[k]: pd.Series(v) for k, v in bundle.metadata["parameters"].items()}), width="stretch")
    for key in ("as_of", "prior_origin", "prior_recorded_at", "target_date", "futcode", "observed_price"):
        rows[key] = snapshot[key]
    for key, value in snapshot["state"].items():
        rows[key] = value
    st.download_button("Download daily forecast CSV", rows.to_csv(index=False),
                       f"{bundle.metadata['dataset_id']}_daily_{snapshot['as_of']}.csv", "text/csv")


def main():
    st.set_page_config(page_title="Commodity Factors", page_icon="◈", layout="wide")
    st.markdown("""<style>
    .block-container {padding-top: 4rem; max-width: 1450px;}
    [data-testid="stMetric"] {background: #181c27; border: 1px solid #2a2e39; padding: 12px; border-radius: 4px;}
    h1 {letter-spacing: -0.035em;} h2, h3 {letter-spacing: -0.02em;}
    </style>""", unsafe_allow_html=True)
    choices = available_datasets()
    st.sidebar.title("Commodity Factors")
    st.sidebar.caption("Explore curves. Compare forecasts.")
    if not choices:
        st.title("Commodity Factors")
        st.info("Create an analysis bundle to open the dashboard.")
        st.code("python scripts/run_synthetic.py\npython scripts/run_empirical.py", language="bash")
        return
    dataset = st.sidebar.selectbox("Dataset", list(choices))
    directory = choices[dataset]
    try:
        revision = tuple((directory / name).stat().st_mtime_ns
                         for name in ["metadata.json", *[f"{table}.parquet" for table in TABLES]])
        bundle = read_analysis(str(directory), revision)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"The saved analysis could not be loaded: {error}")
        st.info("Regenerate the analysis bundle with the corresponding analysis script.")
        return
    meta = bundle.metadata
    split = st.sidebar.selectbox("Evaluation period", ["test", "validation"], format_func=str.title)
    horizon = st.sidebar.selectbox("Forecast horizon", meta["horizons"],
                                   format_func=lambda value: f"{value} trading day{'s' if value != 1 else ''}")
    rank = st.sidebar.selectbox("Maturity rank", sorted(bundle.panel.maturity_rank.unique()))
    models = st.sidebar.multiselect("Models", list(MODEL_LABELS), default=list(MODEL_LABELS),
                                    format_func=lambda value: MODEL_LABELS[value])
    selected = bundle.predictions[(bundle.predictions.split == split) & (bundle.predictions.horizon == horizon)]
    if selected.empty:
        st.info("No forecasts are available for this evaluation period and horizon.")
        return
    first, last = selected.date_.min().date(), selected.date_.max().date()
    window = st.sidebar.date_input("Evaluation window", value=(first, last), min_value=first, max_value=last,
                                   key=f"window-{meta['dataset_id']}-{split}-{horizon}")
    if len(window) != 2:
        st.info("Choose both a start and end date for the evaluation window.")
        return
    selected = selected[(selected.date_ >= pd.Timestamp(window[0])) & (selected.date_ <= pd.Timestamp(window[1]))]
    st.sidebar.divider()
    st.sidebar.caption(f"Training ends {meta['train_end']}. Parameters remain fixed during validation and test.")

    st.caption("COMMODITY RESEARCH / FUTURES TERM STRUCTURE")
    st.title("Commodity Factors")
    st.caption(dataset)
    st.write("See how random walk, one-factor, and two-factor models describe and predict a futures curve.")
    if meta["dataset_id"] == "synthetic":
        st.info("Synthetic demo · These prices are generated. Results demonstrate the workflow, not historical market performance.")
    else:
        st.caption(f"{meta['source']} · Settlement prices · {meta['price_unit']}")
    if selected.empty or not models:
        st.info("Select at least one model and an evaluation window containing trading dates.")
        return
    view = st.radio("View", ["Market overview", "Forecast comparison", "Performance", "Model exploration", "Daily forecast"],
                    horizontal=True, label_visibility="collapsed")
    st.divider()
    if view == "Daily forecast":
        daily_forecasts(bundle, rank, models)
        return
    metrics = summarize_predictions(selected)
    overall = metrics[metrics.maturity_rank == 0].set_index("model")
    columns = st.columns(4)
    columns[0].metric("Scored dates", f"{selected.date_.nunique():,}")
    for column, model in zip(columns[1:], MODEL_LABELS):
        column.metric(f"{MODEL_LABELS[model]} · log RMSE", f"{overall.loc[model, 'rmse_log']:.4f}")
    st.caption(f"{split.title()} · {horizon}-day horizon · All {meta['quality']['contracts_per_day']} maturity ranks")
    if view == "Market overview":
        overview(bundle, rank)
    elif view == "Forecast comparison":
        forecasts(bundle, selected, rank, models, horizon)
    elif view == "Performance":
        performance(bundle, selected, models, split, horizon, window)
    else:
        exploration(bundle, split)
    with st.expander("Data and model assumptions"):
        for assumption in meta["assumptions"]:
            st.write(f"• {assumption}")


if __name__ == "__main__":
    main()
