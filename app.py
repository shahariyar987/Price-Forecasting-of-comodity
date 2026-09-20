"""Bangladesh Commodity Price Predictor - Streamlit user interface.

This file only builds the screens. All calculations live in data_tools.py.
"""
import hmac
import os
import re

import altair as alt
import pandas as pd
import streamlit as st

import data_tools as dt

APP_VERSION = "1.1.0"
FORECAST_DISCLAIMER = ("⚠️ AI can make mistakes. This forecast is only an estimate from a machine-learning model "
                       "trained on limited data, so don't believe it blindly. Please check with real market prices "
                       "before making any decision.")
ESTIMATE_DISCLAIMER = ("⚠️ AI can make mistakes. This estimate is only based on limited data, so don't believe it "
                       "blindly. Please check with real market prices before making any decision.")

st.set_page_config(page_title="Bangladesh Commodity Price Predictor", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1200px;}
#MainMenu, footer {visibility: hidden;}
.hero {background: linear-gradient(135deg, #0B3D2E 0%, #14705A 100%); padding: 1.5rem 1.8rem;
       border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {color: #FFFFFF !important; font-size: 1.85rem; margin: 0 0 0.25rem 0; padding: 0;}
.hero p {color: #D5EDE4; margin: 0; font-size: 1rem;}
.app-footer {text-align: center; color: #6B7C76; font-size: 0.8rem; margin-top: 2rem;
             padding-top: 1rem; border-top: 1px solid #E3ECE8;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data (cached, refreshed when files change)
@st.cache_data(show_spinner=False)
def get_monthly(signature):
    return dt.load_effective_monthly()


@st.cache_data(show_spinner=False)
def get_districts(signature):
    return dt.load_district_prices()


@st.cache_resource(show_spinner=False)
def get_model(signature):
    return dt.load_model()


def get_admin_password():
    """The admin password comes from .streamlit/secrets.toml or the ADMIN_PASSWORD environment variable."""
    try:
        return st.secrets["ADMIN_PASSWORD"]
    except Exception:
        return os.environ.get("ADMIN_PASSWORD")


missing = [p for p in (dt.GENERATED_PATH, dt.DISTRICT_PATH) if not os.path.exists(p)]
if missing:
    st.error("Some required data files are missing: " + ", ".join(missing) +
             ". Run notebooks/01_data_preparation.ipynb first.")
    st.stop()

monthly_df = get_monthly(dt.data_signature())
district_df = get_districts(dt.data_signature())
try:
    model, model_features = get_model(dt.model_signature())
except ValueError as error:
    st.error(str(error))
    st.stop()


def index_of(options, wanted):
    return options.index(wanted) if wanted in options else 0


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "**Forecast** predicts the retail price of a commodity in a district for the coming months.\n\n"
        "**Cross-district** estimates the price in one district from the known price in another.\n\n"
        "Prices are in Bangladeshi Taka per kilogram. All numbers are estimates and can be wrong, and the "
        "further ahead a forecast goes, the less reliable it is."
    )
    st.caption(f"Version {APP_VERSION}")

# ---------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>🌾 Bangladesh Commodity Price Predictor</h1>
  <p>Forecast retail prices and estimate prices across districts for chicken, rice and fish.</p>
</div>
""", unsafe_allow_html=True)

tab_forecast, tab_cross, tab_admin = st.tabs([
    ":material/trending_up: Forecast",
    ":material/compare_arrows: Cross-district",
    ":material/admin_panel_settings: Admin",
], key="main_tabs")

# ================================================================ TAB 1: FORECAST
with tab_forecast:
    left, right = st.columns([1, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Forecast settings")
            locations = sorted(monthly_df["location"].unique())
            loc1 = st.selectbox("District", locations, index=index_of(locations, "Dhaka"), key="loc1")

            counts = monthly_df[monthly_df["location"] == loc1].groupby("commodity").size()
            commodities = sorted(counts[counts >= 4].index.tolist())
            if commodities:
                comm1 = st.selectbox("Commodity", commodities, index=index_of(commodities, "Broiler chicken"), key="comm1")
            else:
                comm1 = None
                st.warning(f"No commodity has enough data in {loc1} to forecast.")

            months_ahead = st.slider("Months to forecast", 1, 18, 6, help="The forecast starts next month.")

    with right:
        result = None
        if comm1:
            try:
                result = dt.forecast_series(monthly_df, model, model_features, loc1, comm1, months_ahead)
            except ValueError as error:
                st.error(str(error))

        if result is None:
            st.info("Choose a district and a commodity to see the forecast.")
        else:
            forecast, history = result["forecast"], result["history"]
            last_price = float(history["price_per_kg"].iloc[-1])
            first, last = forecast.iloc[0], forecast.iloc[-1]

            m1, m2 = st.columns(2)
            m1.metric(f"Next month ({first['date']:%b %Y})", f"{first['price']:.2f} Tk/kg",
                      f"{(first['price'] / last_price - 1) * 100:+.1f}% vs now", delta_color="off", border=True)
            m2.metric(f"In {months_ahead} month(s) ({last['date']:%b %Y})", f"{last['price']:.2f} Tk/kg",
                      f"{(last['price'] / last_price - 1) * 100:+.1f}% vs now", delta_color="off", border=True)

            past = history.rename(columns={"price_per_kg": "price"}).assign(kind="History")
            bridge = past.tail(1).assign(kind="Forecast")            # joins the two lines
            chart_data = pd.concat([past, bridge, forecast.assign(kind="Forecast")], ignore_index=True)
            chart = (
                alt.Chart(chart_data)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")),
                    y=alt.Y("price:Q", title="Tk per kg", scale=alt.Scale(zero=False)),
                    color=alt.Color("kind:N", legend=alt.Legend(title=None, orient="top"),
                                    scale=alt.Scale(domain=["History", "Forecast"], range=["#14705A", "#E08A00"])),
                    strokeDash=alt.StrokeDash("kind:N", legend=None,
                                              scale=alt.Scale(domain=["History", "Forecast"], range=[[1, 0], [6, 4]])),
                    tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"),
                             alt.Tooltip("price:Q", title="Tk/kg", format=".2f"), alt.Tooltip("kind:N", title="")],
                )
                .properties(height=330)
            )
            st.altair_chart(chart, width="stretch")

            table = pd.DataFrame({"Month": forecast["date"].dt.strftime("%Y-%m"),
                                  "Predicted price (Tk/kg)": forecast["price"].round(2)})
            with st.expander("Forecast table"):
                st.dataframe(table, hide_index=True)
                file_name = re.sub(r"[^A-Za-z0-9]+", "_", f"forecast_{loc1}_{comm1}") + ".csv"
                st.download_button("Download forecast (CSV)", table.to_csv(index=False), file_name=file_name,
                                   mime="text/csv")

            if result["source"] != "uploaded":
                st.caption("Based on estimated monthly prices, not on observed monthly market data.")
            st.warning(FORECAST_DISCLAIMER)

            if st.session_state.get("is_admin", False):
                with st.expander("Details (administrators only)"):
                    st.write(f"Last known month: **{result['last_known']:%b %Y}** ({last_price:.2f} Tk/kg) · "
                             f"data basis: **{result['source']}**")
                    history_table = result["history"].assign(
                        date=result["history"]["date"].dt.strftime("%Y-%m"),
                        price_per_kg=result["history"]["price_per_kg"].round(2))
                    st.dataframe(history_table, hide_index=True)

# ================================================================ TAB 2: CROSS-DISTRICT
with tab_cross:
    with st.container(border=True):
        st.subheader("Estimate the price in another district")
        locations2 = sorted(district_df["location"].unique())
        c1, c2 = st.columns(2)
        source_loc = c1.selectbox("Source district (known price)", locations2,
                                  index=index_of(locations2, "Chattogram"), key="src")
        target_loc = c2.selectbox("Target district (price to estimate)", locations2,
                                  index=index_of(locations2, "Dhaka"), key="tgt")

        commodities2 = dt.common_commodities(district_df, source_loc, target_loc)
        estimate = None
        if source_loc == target_loc:
            st.warning("Please choose two different districts.")
        elif not commodities2:
            st.warning(f"{source_loc} and {target_loc} have no commodity in common.")
        else:
            c3, c4 = st.columns(2)
            comm2 = c3.selectbox("Commodity", commodities2, index=index_of(commodities2, "Broiler chicken"), key="comm2")
            try:
                estimate = dt.cross_district_estimate(district_df, source_loc, target_loc, comm2)
            except ValueError as error:
                st.warning(str(error))
            else:
                source_price = c4.number_input(f"Current price in {source_loc} (Tk/kg)", min_value=0.0,
                                               value=round(estimate["typical_source"], 2), step=1.0)

    if estimate is not None:
        typical = estimate["typical_source"]
        if abs(source_price / typical - 1) > 0.5:
            st.warning(f"The usual price of {comm2} in {source_loc} is about {typical:.2f} Tk/kg. "
                       "Please double-check the price you entered.")

        st.metric(f"Estimated price in {target_loc}", f"{source_price * estimate['ratio']:.2f} Tk/kg", border=True)
        st.warning(ESTIMATE_DISCLAIMER)

        if st.session_state.get("is_admin", False):
            with st.expander("Details (administrators only)"):
                st.markdown(f"For every year with prices in **both** districts, the price in {target_loc} is divided "
                            f"by the price in {source_loc}. The average of these ratios "
                            f"(**{estimate['ratio']:.3f}**) is multiplied by the price entered.")
                st.dataframe(pd.DataFrame({
                    "Year": [str(y) for y in estimate["table"].columns],
                    f"{source_loc} (Tk/kg)": estimate["table"].loc[source_loc].round(2).values,
                    f"{target_loc} (Tk/kg)": estimate["table"].loc[target_loc].round(2).values,
                    "Ratio": estimate["yearly_ratio"].round(3).values,
                }), hide_index=True)

# ================================================================ TAB 3: ADMIN
with tab_admin:
    admin_password = get_admin_password()

    if not admin_password:
        st.info("Admin access is not set up yet. Create the file `.streamlit/secrets.toml` and add this line, "
                "with your own password:\n\n`ADMIN_PASSWORD = \"your-password\"`\n\nThen restart the app. "
                "See README.md for details.")

    elif not st.session_state.get("is_admin", False):
        _, middle, _ = st.columns([1, 1.2, 1])
        with middle:
            with st.container(border=True):
                st.subheader("Administrator login")
                with st.form("login_form"):
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Unlock", type="primary")
                if submitted:
                    if hmac.compare_digest(password.encode(), admin_password.encode()):
                        st.session_state["is_admin"] = True
                        st.rerun()
                    else:
                        st.error("Wrong password.")

    else:
        head_left, head_right = st.columns([5, 1])
        head_left.subheader("Data and model administration")
        if head_right.button("Lock", key="lock_admin"):
            st.session_state["is_admin"] = False
            st.rerun()

        flash = st.session_state.pop("flash", None)        # message left by the previous action
        if flash:
            getattr(st, flash[0])(flash[1])

        t_status, t_upload, t_train, t_format = st.tabs(["Status", "Upload data", "Retrain model", "File format"],
                                                        key="admin_tabs")

        # ---------------- status
        with t_status:
            if st.button("Undo last upload"):
                if dt.undo_last_upload():
                    st.session_state["flash"] = ("success", "The previous version of the uploaded data was restored.")
                else:
                    st.session_state["flash"] = ("info", "There is nothing to undo.")
                st.rerun()
            summary = dt.dataset_summary(monthly_df)
            status = dt.uploaded_series_status()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Districts", summary["districts"], border=True)
            s2.metric("Price series", summary["series"], border=True)
            s3.metric("Latest data month", summary["latest_month"].strftime("%b %Y"), border=True)
            s4.metric("Series using uploaded data",
                      int((status["status"] == "using uploaded data").sum()) if not status.empty else 0, border=True)
            if status.empty:
                st.info("No uploaded data yet: every series uses the generated monthly data.")
            else:
                st.dataframe(status, hide_index=True)

        # ---------------- upload
        with t_upload:
            allow_new = st.checkbox("Allow NEW district / commodity names (only if you are sure they are correct)")
            uploaded = st.file_uploader("Choose a CSV file", type=["csv"],
                                        key=f"upload_{st.session_state.get('upload_round', 0)}")

            if uploaded is not None:
                uploaded.seek(0)
                try:
                    raw = pd.read_csv(uploaded, dtype=str, encoding="utf-8-sig")
                except Exception as error:
                    raw = None
                    st.error(f"This file could not be read as a CSV: {error}")

                if raw is not None:
                    known_locs, known_coms, medians = dt.known_names()
                    accepted, rejected, merged, error = dt.validate_upload(raw, known_locs, known_coms, medians, allow_new)
                    if error:
                        st.error(error)
                    elif len(raw) == 0:
                        st.warning("The file has no data rows.")
                    else:
                        u1, u2, u3 = st.columns(3)
                        u1.metric("Rows in file", len(raw), border=True)
                        u2.metric("Accepted", len(accepted), border=True)
                        u3.metric("Rejected", len(rejected), border=True)
                        if merged > 0:
                            st.info(f"{merged} rows were averaged with other rows of the same district, commodity and month.")
                        if len(rejected) > 0:
                            st.warning("These rows were rejected. Fix them in your file and upload it again, "
                                       "or save only the accepted rows.")
                            st.dataframe(rejected, hide_index=True)
                        if len(accepted) > 0:
                            st.write("Accepted rows (preview):")
                            st.dataframe(accepted.head(20), hide_index=True)
                            if st.button("Save accepted rows to the database", type="primary"):
                                added, updated = dt.save_upload(accepted)
                                st.session_state["flash"] = ("success", f"Saved: {added} new rows added, "
                                                             f"{updated} existing rows updated. A backup was made first.")
                                st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
                                st.rerun()

        # ---------------- retrain
        with t_train:
            st.write("Training first tests the new model on the latest months it has not seen. "
                     "Nothing is replaced until you press the second button, and the old model is backed up first.")
            if st.button("Train a new model (test only)"):
                with st.spinner("Training..."):
                    try:
                        st.session_state["candidate"] = dt.train_candidate(dt.load_effective_monthly())
                    except ValueError as error:
                        st.session_state.pop("candidate", None)
                        st.error(str(error))

            candidate = st.session_state.get("candidate")
            if candidate is not None:
                m = candidate["metrics"]
                q1, q2, q3 = st.columns(3)
                q1.metric("New model error (MAE)", f"{m['mae_model']:.2f} Tk/kg", border=True)
                q2.metric("Baseline error (MAE)", f"{m['mae_baseline']:.2f} Tk/kg", border=True)
                q3.metric("Test rows using real data", f"{m['real_test_rows']} of {m['test_rows']}", border=True)
                st.caption(f"Trained on {m['rows']:,} rows from {m['series']} series and tested on the latest "
                           f"{m['test_months']} months. MAE = the average mistake on months the model has not seen "
                           "(lower is better). Baseline = 'next month = this month'.")
                if m["real_test_rows"] < 20:
                    st.warning("Very few test rows use real data, so this test mostly measures the generated data. "
                               "A pass here does not prove the model is good on real prices.")
                if m["passed"]:
                    st.success("The new model makes smaller mistakes than the simple baseline.")
                    if st.button("Replace the current model with this new model", type="primary"):
                        dt.install_model(candidate)
                        st.session_state.pop("candidate", None)
                        st.session_state["flash"] = ("success", "Done. The Forecast tab now uses the new model. "
                                                     "The old model was saved in models/backups.")
                        st.rerun()
                else:
                    st.error("The new model did NOT beat the simple baseline, so it will not be used.")

            if os.path.exists(dt.TRAINING_LOG):
                st.write("Training history")
                st.dataframe(pd.read_csv(dt.TRAINING_LOG), hide_index=True)

        # ---------------- file format
        with t_format:
            st.write("Upload a **CSV** file with exactly these 4 columns (any order):")
            st.table(pd.DataFrame({
                "Column": ["date", "location", "commodity", "price_per_kg"],
                "What to put": ["First day of the month", "District name", "Commodity name", "Price in Tk per kg"],
                "Example": ["2026-08-01", "Dhaka", "Broiler chicken", "181.50"],
            }))
            st.code("date,location,commodity,price_per_kg\n"
                    "2026-07-01,Dhaka,Broiler chicken,180.20\n"
                    "2026-08-01,Dhaka,Broiler chicken,181.50\n"
                    "2026-08-01,Chattogram,Broiler chicken,168.30", language="text")
            st.markdown(
                "- One row = one month of one commodity in one district.\n"
                "- Dates must look like `2026-08-01` (or `2026-08`). Formats like `8/1/2026` are rejected because day and month can be swapped.\n"
                "- Prices are numbers above 0 in Tk/kg. Prices far away from the usual price of that commodity are rejected.\n"
                "- District and commodity names must match the app's names (capital letters do not matter). Wrong spellings are rejected with a suggestion.\n"
                "- Several rows for the same district, commodity and month (for example different markets) are averaged into one.\n"
                "- Future months are rejected.\n"
                f"- A district + commodity switches from generated to **real** data only after at least **{dt.MIN_REAL_MONTHS} consecutive months** have been uploaded."
            )
            st.download_button("Download empty template", data="date,location,commodity,price_per_kg\n",
                               file_name="upload_template.csv", mime="text/csv")

# ---------------------------------------------------------------- footer
st.markdown(f'<div class="app-footer">Bangladesh Commodity Price Predictor · v{APP_VERSION} · '
            'Predictions are estimates and may be wrong.</div>', unsafe_allow_html=True)
