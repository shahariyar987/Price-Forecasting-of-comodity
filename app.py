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

APP_VERSION = "2.1.0"
FORECAST_DISCLAIMER = ("⚠️ AI can make mistakes. This forecast is only an estimate from a machine-learning model "
                       "trained on limited data, so don't believe it blindly. Please check with real market prices "
                       "before making any decision.")
ESTIMATE_DISCLAIMER = ("⚠️ AI can make mistakes. This estimate is only based on limited data, so don't believe it "
                       "blindly. Please check with real market prices before making any decision.")

st.set_page_config(page_title="Bangladesh Commodity Price Predictor", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- design tokens + chart theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #182420; --paper: #F7F4EC; --paper-raised: #FFFFFF;
  --forest-deep: #0B3D2E; --forest: #14705A; --harvest: #C97A1A; --river: #1F6FB2;
  --line: #E1DAC5; --muted: #5B6B63;
}
.stApp { background: var(--paper); }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }
h1, h2, h3, .headline { font-family: 'Fraunces', serif; font-weight: 600; letter-spacing: -0.01em; }
/* Extra top clearance: Streamlit's own header bar (and, when the sidebar is collapsed, its floating
   expand-sidebar arrow) occupies this space above the normal page flow - without it, our own topbar sits
   directly underneath and the two visually collide. */
.block-container {padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1200px;}
#MainMenu, footer {visibility: hidden;}
[data-testid="stSidebar"] { background: var(--paper-raised); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h2 { font-family: 'Fraunces', serif; }

div[data-testid="stMetric"] { background: var(--paper-raised); border: 1px solid var(--line); border-radius: 10px;
                              padding: 0.75rem 0.9rem; transition: box-shadow 0.15s ease, transform 0.15s ease; }
div[data-testid="stMetric"]:hover { box-shadow: 0 3px 10px rgba(11, 61, 46, 0.08); transform: translateY(-1px); }
div[data-testid="stMetricLabel"] { color: var(--muted); }
button[kind="primary"] { background: var(--forest); border-color: var(--forest); transition: background 0.15s ease, transform 0.1s ease; }
button[kind="primary"]:hover { background: var(--forest-deep); border-color: var(--forest-deep); transform: translateY(-1px); }
button { transition: transform 0.1s ease; }
button:active { transform: scale(0.98); }

.topbar { display: flex; justify-content: space-between; align-items: center;
         padding: 0.7rem 1.1rem; background: var(--paper-raised); border: 1px solid var(--line);
         border-radius: 12px; margin-bottom: 1.1rem; }
.topbar .brand { font-family: 'Fraunces', serif; font-size: 1.25rem; color: var(--forest-deep); }
.topbar .brand span { color: var(--muted); font-family: 'IBM Plex Sans', sans-serif; font-size: 0.8rem;
                      margin-left: 0.5rem; }
.identity-chip { color: var(--muted); font-size: 0.92rem; }
.identity-chip b { color: var(--ink); }

.app-footer { text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 2rem;
             padding-top: 1rem; border-top: 1px solid var(--line); }

/* One deliberate entrance moment on the landing page only - not repeated anywhere else in the app. */
@keyframes fade-up { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.landing-hero { animation: fade-up 0.5s ease-out; }

/* Landing feature cards lift slightly on hover, signalling "look here" without being asked to click anything. */
[data-testid="stVerticalBlockBorderWrapper"]:has(.feature-card) { transition: transform 0.15s ease, box-shadow 0.15s ease; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.feature-card):hover { transform: translateY(-3px); box-shadow: 0 6px 16px rgba(11, 61, 46, 0.10); }
</style>
""", unsafe_allow_html=True)


def _market_chart_theme():
    return alt.theme.ThemeConfig({
        "background": "transparent",
        "font": "IBM Plex Sans, sans-serif",
        "title": {"font": "Fraunces, serif", "fontSize": 15, "color": "#182420"},
        "axis": {"labelColor": "#5B6B63", "titleColor": "#5B6B63", "gridColor": "#E1DAC5",
                 "domainColor": "#CFC7AD", "tickColor": "#CFC7AD", "labelFontSize": 11.5},
        "legend": {"labelColor": "#182420", "titleColor": "#182420"},
        "view": {"stroke": "transparent"},
        "range": {"category": ["#14705A", "#C97A1A", "#1F6FB2", "#8A5A44"]},
    })


alt.theme.register("market_report", enable=True)(_market_chart_theme)


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


# ---------------------------------------------------------------- landing page + login
def render_landing():
    total_districts = monthly_df["location"].nunique()
    total_commodities = monthly_df["commodity"].nunique()
    latest_month = monthly_df["date"].max().strftime("%B %Y")

    st.markdown(f"""
    <div class="landing-hero" style="background: linear-gradient(135deg, var(--forest-deep) 0%, var(--forest) 100%);
                border-radius: 16px; padding: 3rem 2.5rem; margin-top: 1rem; margin-bottom: 1.5rem;">
      <div class="headline" style="color: #fff; font-size: 2.3rem; line-height: 1.2; max-width: 640px;">
        Know where commodity prices in Bangladesh are headed.
      </div>
      <p style="color: #D5EDE4; font-size: 1.05rem; max-width: 560px; margin-top: 0.9rem;">
        Forecasts, district comparisons and a buying-budget planner for chicken, rice and fish -
        built from real Department of Agricultural Marketing data.
      </p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("Districts tracked", total_districts, border=True)
    s2.metric("Commodities tracked", total_commodities, border=True)
    s3.metric("Latest data month", latest_month, border=True)

    st.write("")
    f1, f2, f3, f4 = st.columns(4)
    with f1.container(border=True):
        st.markdown('<span class="feature-card"></span>', unsafe_allow_html=True)
        st.markdown("**:material/trending_up: Forecast**")
        st.caption("See where a commodity's price is headed in any district, months ahead.")
    with f2.container(border=True):
        st.markdown('<span class="feature-card"></span>', unsafe_allow_html=True)
        st.markdown("**:material/compare_arrows: Cross-district**")
        st.caption("Know a price in one district? Estimate it in another.")
    with f3.container(border=True):
        st.markdown('<span class="feature-card"></span>', unsafe_allow_html=True)
        st.markdown("**:material/savings: Budget planner**")
        st.caption("Given a budget, see what might be worth buying and holding.")
    with f4.container(border=True):
        st.markdown('<span class="feature-card"></span>', unsafe_allow_html=True)
        st.markdown("**:material/edit_note: Submit a price**")
        st.caption("Seen a price locally? Add it and earn a point.")

    st.write("")
    _, mid, _ = st.columns([1, 1, 1])
    if mid.button("Get started", type="primary", width="stretch"):
        st.session_state["stage"] = "login"
        st.rerun()


def render_login():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="headline" style="font-size: 1.5rem;">Continue with your name or phone</div>',
                        unsafe_allow_html=True)
            st.caption("No password. If this is new, it becomes your account; if you've used it before, "
                      "you're back in the same one - along with any points you've earned.")
            with st.form("login_form_main"):
                identifier_input = st.text_input("Name or phone number", label_visibility="collapsed",
                                                 placeholder="e.g. Rafi Uddin or 01710000000")
                go = st.form_submit_button("Continue", type="primary", width="stretch")
            if go:
                try:
                    identifier, is_new = dt.login_or_register(identifier_input)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.session_state["user_id"] = identifier
                    st.session_state["stage"] = "app"
                    st.session_state["welcome"] = "new" if is_new else "back"
                    st.rerun()
        if st.button("← Back"):
            st.session_state["stage"] = "landing"
            st.rerun()


if "stage" not in st.session_state:
    st.session_state["stage"] = "landing"

if st.session_state["stage"] == "landing":
    render_landing()
    st.stop()

if st.session_state["stage"] == "login":
    render_login()
    st.stop()

user_id = st.session_state["user_id"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "**Forecast** predicts the retail price of a commodity in a district for the coming months.\n\n"
        "**Cross-district** estimates the price in one district from the known price in another.\n\n"
        "**Budget planner** ranks what might be worth buying given a budget and a time horizon.\n\n"
        "**Submit a price** lets anyone contribute a price and earn a point once it's added.\n\n"
        "Prices are in Bangladeshi Taka per kilogram. All numbers are estimates and can be wrong, and the "
        "further ahead a forecast goes, the less reliable it is."
    )
    st.caption(f"Version {APP_VERSION}")
    if st.button("Log out", width="stretch"):
        for key in ("stage", "user_id", "is_admin", "welcome"):
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------------- top bar
welcome = st.session_state.pop("welcome", None)
if welcome == "new":
    st.success(f"Account created. Welcome, {user_id}.")
    st.balloons()          # a one-time moment, only ever shown once per new account - never repeated on ordinary logins
elif welcome == "back":
    st.success(f"Welcome back, {user_id}.")

status = dt.contributor_status(user_id)
top_left, top_right = st.columns([3, 2])
with top_left:
    st.markdown(f'<div class="topbar"><div class="brand">🌾 Bangladesh Commodity Price Predictor'
               f'<span>v{APP_VERSION}</span></div></div>', unsafe_allow_html=True)
with top_right:
    st.markdown(f'<div class="topbar"><div class="identity-chip">👤 <b>{user_id}</b> · '
               f'{status["free_remaining"]} free run(s) · {status["points_balance"]} point(s)</div></div>',
               unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs
tab_forecast, tab_cross, tab_budget, tab_submit, tab_admin = st.tabs([
    ":material/trending_up: Forecast",
    ":material/compare_arrows: Cross-district",
    ":material/savings: Budget planner",
    ":material/edit_note: Submit a price",
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
                                    scale=alt.Scale(domain=["History", "Forecast"], range=["#14705A", "#C97A1A"])),
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

                    if st.checkbox("View all data (every district and commodity)", key="forecast_view_all"):
                        st.caption(f"The full monthly dataset the Forecast tab draws from - {len(monthly_df):,} rows.")
                        display_all = monthly_df.assign(date=monthly_df["date"].dt.strftime("%Y-%m"))
                        st.dataframe(display_all, hide_index=True, height=320)

    if comm1:
        with st.container(border=True):
            st.subheader(f"{comm1}: every district compared")
            by_district = dt.commodity_price_by_district(monthly_df, comm1)
            by_district["Selected"] = by_district["location"].apply(lambda l: "This district" if l == loc1 else "Other")

            base = alt.Chart(by_district).encode(
                y=alt.Y("location:N", sort="-x", title=None),
                x=alt.X("price_per_kg:Q", title="Tk per kg"),
            )
            stems = base.mark_rule(color="#D9D1B8", strokeWidth=1.5)
            points = base.mark_circle(size=110).encode(
                color=alt.Color("Selected:N", legend=None,
                                scale=alt.Scale(domain=["This district", "Other"], range=["#C97A1A", "#14705A"])),
                tooltip=[alt.Tooltip("location:N", title="District"),
                         alt.Tooltip("price_per_kg:Q", title="Tk/kg", format=".2f")],
            )
            st.altair_chart((stems + points).properties(height=max(220, 22 * len(by_district))), width="stretch")
            st.caption(f"Latest available price of {comm1} in every district (amber = {loc1}).")

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

        trend = estimate["table"].reset_index().melt(id_vars="location", var_name="Year", value_name="price_per_kg")
        trend["Year"] = trend["Year"].astype(str)
        line = (
            alt.Chart(trend)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("Year:O", title=None),
                y=alt.Y("price_per_kg:Q", title="Tk per kg", scale=alt.Scale(zero=False)),
                color=alt.Color("location:N", legend=alt.Legend(title=None, orient="top"),
                                scale=alt.Scale(domain=[source_loc, target_loc], range=["#14705A", "#C97A1A"])),
                tooltip=[alt.Tooltip("location:N", title="District"), alt.Tooltip("Year:O", title="Year"),
                         alt.Tooltip("price_per_kg:Q", title="Tk/kg", format=".2f")],
            )
            .properties(height=260)
        )
        st.altair_chart(line, width="stretch")
        st.caption(f"Real yearly average price of {comm2} in {source_loc} and {target_loc} "
                  "(only years where both have a price).")

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

                if st.checkbox("View all data (every district and commodity)", key="cross_view_all"):
                    st.caption(f"The full real yearly-average dataset the Cross-district tab draws from - "
                              f"{len(district_df):,} rows.")
                    st.dataframe(district_df, hide_index=True, height=320)

# ================================================================ TAB 3: BUDGET PLANNER
with tab_budget:
    st.info("This ranks predicted price movement to help decide what might be worth buying now - it is "
           "not financial advice. See the disclaimer below before acting on it.")

    with st.container(border=True):
        st.subheader("What could be worth buying and holding?")
        b1, b2, b3 = st.columns(3)
        budget = b1.number_input("Budget (Tk)", min_value=1.0, value=500.0, step=50.0)
        horizon = b2.slider("Time horizon (months)", 1, 18, 6, key="budget_horizon")
        locations4 = ["All districts"] + sorted(monthly_df["location"].unique())
        loc4 = b3.selectbox("District", locations4, index=index_of(locations4, "Dhaka"), key="loc4")
        if loc4 == "All districts":
            st.caption("Scanning every district can take up to about 30 seconds.")

        run = st.button("Find opportunities", type="primary")

    if run:
        try:
            credit_used = dt.consume_budget_plan_credit(user_id)
        except ValueError as error:
            st.session_state.pop("budget_opportunities", None)
            st.error(str(error))
        else:
            scan_location = None if loc4 == "All districts" else loc4
            with st.spinner("Forecasting every eligible commodity..."):
                opportunities = dt.budget_opportunities(monthly_df, model, model_features, budget, horizon,
                                                         location=scan_location)
            st.session_state["budget_opportunities"] = opportunities
            new_status = dt.contributor_status(user_id)
            spent = "a free trial" if credit_used == "free_trial" else "1 point"
            st.success(f"Used {spent}. {new_status['free_remaining']} free run(s) and "
                      f"{new_status['points_balance']} point(s) left.")

    opportunities = st.session_state.get("budget_opportunities")
    if opportunities is not None:
        if opportunities.empty:
            st.warning("Nothing affordable was found for this budget, district and horizon. "
                      "Fish is always excluded (it does not store well).")
        else:
            top = opportunities.iloc[0]
            st.metric(f"Best predicted riser: {top['commodity']} ({top['location']})",
                     f"{top['predicted_change_pct']:+.1f}% in {horizon} month(s)",
                     f"{top['current_price']:.2f} -> {top['predicted_price']:.2f} Tk/kg",
                     delta_color="off", border=True)
            st.dataframe(
                opportunities.rename(columns={
                    "location": "District", "commodity": "Commodity", "category": "Category",
                    "current_price": "Now (Tk/kg)", "predicted_price": f"In {horizon}mo (Tk/kg)",
                    "predicted_change_pct": "Predicted change (%)", "kg_affordable_now": "Kg affordable now",
                    "data_basis": "Data basis",
                }),
                hide_index=True,
            )
            if (opportunities["data_basis"] != "uploaded").any():
                st.caption("Rows marked with generated data basis are ranked on the estimated monthly series "
                          "(see Known limitations in the README) - treat the ranking as indicative, not exact.")
    st.warning(FORECAST_DISCLAIMER)

# ================================================================ TAB 4: SUBMIT A PRICE
with tab_submit:
    with st.container(border=True):
        st.subheader("Submit a price you've seen")
        st.write(f"Logged in as **{user_id}**. A price that matches the recent trend for that item is added "
                "immediately and earns you 1 point right away. One that looks like an unusual jump or drop "
                "is held for an administrator to check first - the point comes once it's approved.")
        st.caption(f"Recorded for the current month, **{dt.current_month_start():%B %Y}** - a submission is "
                  "always for the price right now, so this isn't something you can change.")

        locations5, commodities5, medians5 = dt.known_names()
        with st.form("community_form", clear_on_submit=True):
            f1, f2 = st.columns(2)
            sub_location = f1.selectbox("District", locations5, index=index_of(locations5, "Dhaka"))
            sub_commodity = f2.selectbox("Commodity", commodities5,
                                         index=index_of(commodities5, "Broiler chicken"))
            sub_price = st.number_input("Price (Tk per kg)", min_value=0.0, step=1.0)
            submitted5 = st.form_submit_button("Submit", type="primary")

        if submitted5:
            ok, status, info = dt.submit_community_price(dt.current_month_start(), sub_location, sub_commodity,
                                                          sub_price, user_id, monthly_df, locations5,
                                                          commodities5, medians5)
            if not ok:
                st.error(f"Not saved: {info}")
            elif status == "auto_approved":
                st.success("Added - this matched the recent price for this item. You earned 1 point.")
            else:
                st.warning(f"Held for review: {info['reason']}. An administrator will check it before it's "
                          "added - you'll get your point once they approve it.")

# ================================================================ TAB 5: ADMIN
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

        pending_count = len(dt.pending_community_submissions())
        community_label = f"Community submissions ({pending_count})" if pending_count else "Community submissions"
        t_status, t_stats, t_results, t_upload, t_community, t_train, t_format = st.tabs(
            ["Status", "Statistics", "Results", "Upload data", community_label, "Retrain model", "File format"],
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

        # ---------------- statistics
        with t_stats:
            stats = dt.commodity_stats(monthly_df)
            cat_sum = dt.category_summary(stats)

            st.write("By category")
            k1, k2, k3 = st.columns(3)
            for col, (_, row) in zip((k1, k2, k3), cat_sum.iterrows()):
                col.metric(f"{row['category']} ({int(row['commodities'])} commodities)",
                          f"{row['avg_price_2026']:.2f} Tk/kg",
                          f"{row['avg_pct_change_2025_2026']:+.1f}% vs 2025", delta_color="off", border=True)

            st.write("By commodity")
            st.dataframe(
                stats.rename(columns={
                    "commodity": "Commodity", "category": "Category", "avg_2025": "2025 avg (Tk/kg)",
                    "avg_2026": "2026 avg (Tk/kg)", "pct_change_2025_2026": "Change 2025->2026 (%)",
                    "districts_covered": "Districts covered", "latest_month_avg_price": "Latest month avg (Tk/kg)",
                }),
                hide_index=True,
            )
            st.caption("2025/2026 averages are the real DAM yearly figures. \"Latest month avg\" is the most "
                      "recent month in the data the Forecast tab uses (generated unless real data was uploaded).")

        # ---------------- results
        with t_results:
            st.subheader("Current model accuracy")
            st.caption("Tests the model that is live right now on the latest months it was not trained to "
                      "predict - this is separate from 'Retrain model', which tries training a brand new one.")
            try:
                acc = dt.evaluate_current_model(monthly_df, model, model_features)
            except ValueError as error:
                st.info(str(error))
            else:
                r1, r2, r3 = st.columns(3)
                r1.metric("Model error (MAE)", f"{acc['mae_model']:.2f} Tk/kg", border=True)
                r2.metric("Baseline error (MAE)", f"{acc['mae_baseline']:.2f} Tk/kg", border=True)
                r3.metric("Mean error (MAPE)", f"{acc['mape_model']:.1f}%", border=True)
                if acc["beats_baseline"]:
                    st.success(f"Beats the 'next month = this month' baseline, tested on the latest "
                              f"{acc['test_months']} month(s), {acc['test_rows']} rows "
                              f"({acc['real_test_rows']} from real data).")
                else:
                    st.warning("Does not currently beat the simple baseline - consider retraining.")

            if os.path.exists(dt.TRAINING_LOG):
                log = pd.read_csv(dt.TRAINING_LOG)
                st.subheader("Accuracy across retrains")
                chart = (
                    alt.Chart(log.reset_index().rename(columns={"index": "Retrain #"}))
                    .transform_fold(["mae_model", "mae_baseline"], as_=["Series", "MAE"])
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Retrain #:O", title=None),
                        y=alt.Y("MAE:Q", title="Tk/kg"),
                        color=alt.Color("Series:N", legend=alt.Legend(title=None, orient="top")),
                    )
                    .properties(height=240)
                )
                st.altair_chart(chart, width="stretch")

            st.subheader("Public submissions")
            sub_stats = dt.submission_stats()
            if sub_stats["total"] == 0:
                st.info("No public submissions yet.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Added immediately", sub_stats["auto_approved"], border=True)
                c2.metric("Awaiting review", sub_stats["pending"], border=True)
                c3.metric("Approved by admin", sub_stats["approved"], border=True)
                c4.metric("Rejected by admin", sub_stats["rejected"], border=True)
                st.caption(f"Of the ones flagged for review: {sub_stats['moderate']} moderate, "
                          f"{sub_stats['severe']} severe.")

            st.subheader("Top contributors")
            board = dt.points_leaderboard()
            if board.empty:
                st.info("Nobody has earned a point yet.")
            else:
                st.dataframe(board.rename(columns={"contributor": "Contributor", "points_balance": "Points"}),
                            hide_index=True)

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
                            preview = accepted.head(20).assign(date=accepted.head(20)["date"].dt.strftime("%Y-%m"))
                            st.dataframe(preview, hide_index=True)
                            if st.button("Save accepted rows to the database", type="primary"):
                                added, updated = dt.save_upload(accepted)
                                st.session_state["flash"] = ("success", f"Saved: {added} new rows added, "
                                                             f"{updated} existing rows updated. A backup was made first.")
                                st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
                                st.rerun()

        # ---------------- community submissions
        with t_community:
            st.write("Prices that looked consistent with the recent trend for that item were already added "
                    "automatically and the contributor already has their point. Only the ones below were "
                    "flagged as unusual and need a decision. Approving adds it to the uploaded database "
                    f"exactly like a CSV upload row and credits the contributor 1 point - it then counts "
                    f"towards the {dt.MIN_REAL_MONTHS}-consecutive-month rule. Rejecting only marks it "
                    "rejected; no point is given.")
            pending = dt.pending_community_submissions()
            if pending.empty:
                st.info("Nothing needs review right now.")
            else:
                for _, row in pending.iterrows():
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([3, 1, 1])
                        c1.write(f"⚠️ **{row['commodity']}** in **{row['location']}**, "
                                f"{row['date']:%Y-%m}: **{row['price_per_kg']:.2f} Tk/kg** - "
                                f"submitted by **{row['contributor']}** ({row['submitted_at']})")
                        if pd.notna(row.get("reason")):
                            c1.caption(row["reason"].capitalize())
                        if c2.button("Approve", key=f"approve_{row['row_id']}", type="primary"):
                            dt.review_community_submission(int(row["row_id"]), approve=True)
                            st.rerun()
                        if c3.button("Reject", key=f"reject_{row['row_id']}"):
                            dt.review_community_submission(int(row["row_id"]), approve=False)
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
                "What to put": ["Year and month (day is ignored if given)", "District name", "Commodity name", "Price in Tk per kg"],
                "Example": ["2026-08 or 2026-08-01", "Dhaka", "Broiler chicken", "181.50"],
            }))
            st.code("date,location,commodity,price_per_kg\n"
                    "2026-07,Dhaka,Broiler chicken,180.20\n"
                    "2026-08,Dhaka,Broiler chicken,181.50\n"
                    "2026-08,Chattogram,Broiler chicken,168.30", language="text")
            st.markdown(
                "- One row = one month of one commodity in one district. Every price in this app is a **monthly** "
                "price - there is no day-level data anywhere, so the day is not meaningful.\n"
                "- Dates must look like `2026-08` (YYYY-MM) or `2026-08-01` (YYYY-MM-DD). Formats like `8/1/2026` "
                "are rejected because day and month can be swapped. If a day is included, **it is ignored** - "
                "`2026-08-15` and `2026-08-01` are both stored as the same month, `2026-08`.\n"
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
