"""Data, forecasting and model utilities for the Bangladesh Commodity Price Predictor.

The Streamlit app (app.py) only handles the user interface: drawing widgets, showing charts, reading what
the person clicked or typed. Every actual calculation - loading a CSV, checking an upload, training the
model, turning a trained model into a multi-month forecast - lives in this one file instead. Two reasons:
  1) it can be tested on its own (see tests/test_project.py) without needing a running Streamlit app, and
  2) the training notebook (notebooks/02_train_forecast_model.ipynb) calls these same functions, so the
     notebook and the live app can never quietly disagree about how the model is built or used.

This file is organised top to bottom in the order a new reader would probably want to follow it: settings,
login, loading data, statistics, checking an upload, saving an upload, public submissions, points, training
the model, using the model to forecast, the cross-district estimate, and small helpers at the end.
"""
import os          # checking whether a file/folder exists, building file paths, listing a folder's contents
import shutil      # copying files (used for backups)
import difflib     # "did you mean...?" suggestions when a typed district/commodity name is close but wrong
from datetime import datetime     # timestamps written into logs, backups and the points ledger

import json        # reading/writing the small file that lists the model's input column names

import pandas as pd    # tables (DataFrames) - almost everything in this file is a pandas table
import xgboost as xgb  # the machine-learning library that provides the forecasting model itself


# ================================================================ SETTINGS YOU CAN CHANGE
# Everything below is a constant: a fixed value used throughout the file. Keeping them together here (and
# never writing a "magic number" straight into the logic further down) means the whole app's behaviour can
# be retuned by editing a single line in this block, without hunting through every function that uses it.

GENERATED_PATH = "data/processed/monthly_hybrid.csv"        # the generated monthly data (built by notebook 01, never modified afterwards)
UPLOADED_PATH = "data/processed/uploaded_monthly.csv"        # real monthly prices added via the Admin tab or an approved public submission
DISTRICT_PATH = "data/processed/district_prices_clean.csv"   # real YEARLY average price per district (straight from the DAM files)
BACKUP_DIR = "data/backups"                                  # a copy of UPLOADED_PATH is saved here before every change, so "Undo" can restore it

# The trained model is saved in XGBoost's own JSON format instead of Python's generic "pickle" format.
# Two practical reasons: it is plain text, so it survives being zipped/emailed/copied without corruption
# (pickle is a binary format that can break silently), and it can still be read even if the XGBoost version
# that opens it later is not exactly the version that trained it.
MODEL_PATH = "models/model1_forecast.json"          # the trained model itself
FEATURES_PATH = "models/model1_features.json"       # the exact list (and order) of input columns the model expects
MODEL_BACKUP_DIR = "models/backups"                 # the previous model is copied here before a retrain replaces it
TRAINING_LOG = "models/training_log.csv"            # one row per retrain, so accuracy over time can be tracked

REQUIRED_COLUMNS = ["date", "location", "commodity", "price_per_kg"]   # every price record in the app has exactly these four fields
MIN_REAL_MONTHS = 6      # a district+commodity needs this many CONSECUTIVE real months before it stops using generated data
MAX_PRICE_FACTOR = 3     # a price more than this many times above/below the commodity's usual (median) price is rejected outright
MAX_UPLOAD_ROWS = 50000  # safety limit so one CSV upload cannot accidentally overwhelm the app

# Public submissions never write straight into the admin's uploaded database UNLESS they look consistent
# with that specific series' own recent price (see assess_submission below). Ones that look unusual land
# here as "pending" and only count towards anything - including the real-data switchover above - once an
# administrator approves them from the Admin tab.
COMMUNITY_PATH = "data/processed/community_submissions.csv"
COMMUNITY_COLUMNS = ["date", "location", "commodity", "price_per_kg", "contributor", "submitted_at", "status",
                    "reference_price", "pct_change", "severity", "reason"]

# How far a submitted price is allowed to move from that EXACT district+commodity's own last known price
# before an administrator has to look at it. This is direction-aware (a rise and a fall use different
# limits) rather than one flat percentage, because in a real market a price *rising* by a given amount is
# ordinary far more often than it *falling* by that same amount is - a rise can just be demand or a poor
# harvest, while an equally large sudden fall is rarer and more often signals a typo (an extra "0" dropped,
# a decimal point in the wrong place) than a genuine crash. All four numbers are fractions: 0.30 means 30%.
#
#   rise or fall within these limits            -> accepted immediately, no administrator needed
#   rise/fall beyond the "accept" limit          -> held for an administrator ("moderate")
#   rise/fall beyond the "severe" limit as well  -> still held, but labelled "severe" so it stands out
#
# This is a separate, FINER check than MAX_PRICE_FACTOR above: MAX_PRICE_FACTOR compares a price to the
# commodity's overall median across every district (it catches an obviously wrong number, like a decimal
# point slipping and turning 18 Tk/kg into 1800), while this compares a price to that one series' own recent
# level (it catches a number that is perfectly plausible in general but is still a sharp, unexplained move
# for that particular item in that particular district). Admin CSV uploads (Admin > Upload data) skip this
# finer check entirely and go straight in - an administrator is already a trusted user, so there is nothing
# to protect against there.
RISE_ACCEPT_LIMIT = 0.30     # a rise of up to 30% needs no administrator
FALL_ACCEPT_LIMIT = 0.15     # a fall of up to 15% needs no administrator
RISE_SEVERE_LIMIT = 0.50     # a rise of 50% or more is labelled "severe" (still just a label - it is already being held either way)
FALL_SEVERE_LIMIT = 0.30     # a fall of 30% or more is labelled "severe"

# Points: there are no passwords anywhere in this system except the shared Admin password below - a
# "contributor" is identified only by the name/phone number they logged in with (see login_or_register).
# Everyone gets a few free Budget planner runs; after those are used up, one point buys one more run, and a
# point is earned whenever a submitted price is added to the dataset (whether that happens immediately or
# only after an administrator approves it later).
POINTS_PATH = "data/processed/points_ledger.csv"
POINTS_COLUMNS = ["contributor", "event", "points", "time"]
FREE_BUDGET_PLANS = 3

# The "login" is exactly this: a name or phone number that has been seen before logs back into the same
# account (with its points intact); one that has not been seen before creates a brand-new account on the
# spot. There is no password behind it - see the README for exactly what that trade-off means in practice.
USERS_PATH = "data/processed/users.csv"
USERS_COLUMNS = ["identifier", "joined_at"]


def current_month_start():
    """Today's date, rounded down to the 1st of the current month (e.g. 14 Aug 2026 -> 1 Aug 2026).

    Used everywhere something needs to be compared against "now" at month resolution: rejecting
    future-dated rows, deciding which months already exist in the generated data, and so on.
    """
    today = pd.Timestamp.today()
    return pd.Timestamp(today.year, today.month, 1)


# ================================================================ LOGIN (no password - a unique name/phone number)
def load_users():
    """Every account that has ever logged in, as a table with columns identifier, joined_at."""
    if not os.path.exists(USERS_PATH):
        return pd.DataFrame(columns=USERS_COLUMNS)     # nobody has logged in yet - an empty table, not an error
    return pd.read_csv(USERS_PATH)


def login_or_register(identifier):
    """Looks up an account by this exact name/phone number, or creates one if it has never been seen before.

    Because the lookup is an exact text match and an identifier is only ever written to USERS_PATH once,
    one identifier can never end up belonging to two different accounts - that is what "unique" means here.
    It does NOT mean the identity is verified: there is no password, so anyone who types someone else's
    name or phone number logs into that same account (and can spend their points). See the README's
    "Accounts" section for why this trade-off was chosen over a full password-based sign-up.

    Returns a tuple: (identifier, is_new) where is_new is True only the very first time this identifier
    is ever seen.
    """
    identifier = str(identifier).strip()          # trim stray spaces so "Rafi" and "Rafi " are the same account
    if not identifier:
        raise ValueError("Please enter your name or phone number.")

    users = load_users()
    if identifier in set(users["identifier"]):
        return identifier, False                  # already known -> this IS them logging back in

    # Not seen before -> register a new account by appending one row. mode="a" (append) means we never have
    # to re-read and re-write the whole file just to add one person; header=... only writes the column
    # names the very first time the file is created.
    entry = pd.DataFrame([{"identifier": identifier, "joined_at": datetime.now().strftime("%Y-%m-%d %H:%M")}])
    os.makedirs(os.path.dirname(USERS_PATH) or ".", exist_ok=True)
    entry.to_csv(USERS_PATH, mode="a", header=not os.path.exists(USERS_PATH), index=False)
    return identifier, True


# ================================================================ LOADING THE DATA
def load_generated():
    """The generated (synthetic) monthly price series built by notebook 01 from the two yearly averages."""
    df = pd.read_csv(GENERATED_PATH, parse_dates=["date"])   # parse_dates turns the "date" column into real Timestamps, not plain text
    df["source"] = "generated"           # tag every row so later code can always tell generated data from real data
    return df


def load_uploaded():
    """Real monthly prices added so far (Admin CSV uploads plus approved/auto-approved public submissions)."""
    if not os.path.exists(UPLOADED_PATH):
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["uploaded_at"])   # nothing uploaded yet - empty table, not an error
    df = pd.read_csv(UPLOADED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def uploaded_runs():
    """For every uploaded series (one district + one commodity): all of its uploaded rows, and separately
    just its newest unbroken run of consecutive months.

    Why this matters: a district/commodity only "graduates" to using real data once it has MIN_REAL_MONTHS
    consecutive months (see load_effective_monthly below) - a few real months with a gap in the middle do
    not count, because the forecasting model's lag features (lag1/lag2/lag3, see build_training_table) need
    an unbroken monthly sequence to make sense of.
    """
    up = load_uploaded()
    runs = {}
    for (loc, com), g in up.groupby(["location", "commodity"]):
        g = g.sort_values("date")
        month_no = g["date"].dt.year * 12 + g["date"].dt.month     # turns e.g. 2026-03 into a single increasing integer
        # month_no.diff() is 1 exactly where two rows are consecutive months; wherever it is anything else
        # (a gap, or the very first row where diff() is NaN), that is the start of a new "run". cumsum()
        # then numbers each run 0, 1, 2, ... and the LAST number is always the most recent run.
        run_id = (month_no.diff() != 1).cumsum()
        runs[(loc, com)] = (g, g[run_id == run_id.iloc[-1]])
    return runs


def load_effective_monthly():
    """The data the rest of the app actually uses: generated data everywhere, EXCEPT for series that have
    earned real data (see uploaded_runs/MIN_REAL_MONTHS above), which use only their real prices instead.

    This is the single function that Forecast, the Budget planner and model training all call to get "the
    monthly price history" - none of them need to know or care whether a given series is generated or real,
    they just get whichever one is currently the right one to use, tagged in the "source" column.
    """
    gen = load_generated()
    cols = REQUIRED_COLUMNS + ["source"]
    real_parts, real_keys = [], []
    for key, (all_rows, run) in uploaded_runs().items():
        if len(run) >= MIN_REAL_MONTHS:            # this series has "graduated" to real data
            part = run.copy()
            part["source"] = "uploaded"
            real_parts.append(part[cols])
            real_keys.append(key)

    if not real_keys:
        return gen[cols]                            # nothing has graduated yet - just the generated data, unchanged

    # Drop the generated rows for every series that now has real data instead, then glue the real rows on.
    idx = pd.MultiIndex.from_frame(gen[["location", "commodity"]])
    gen_keep = gen[~idx.isin(real_keys)]
    out = pd.concat([gen_keep[cols]] + real_parts, ignore_index=True)
    return out.sort_values(["location", "commodity", "date"]).reset_index(drop=True)


def uploaded_series_status():
    """One row per uploaded series, showing how close it is to the MIN_REAL_MONTHS switchover. Used by the
    Admin > Status tab so an administrator can see progress at a glance."""
    rows = []
    for (loc, com), (all_rows, run) in uploaded_runs().items():
        ready = len(run) >= MIN_REAL_MONTHS
        rows.append({
            "location": loc,
            "commodity": com,
            "months_uploaded": len(all_rows),           # every month ever uploaded for this series, gaps included
            "consecutive_months": len(run),              # just the current unbroken run - this is what actually counts
            "last_month": run["date"].max().strftime("%Y-%m"),
            "status": "using uploaded data" if ready else f"waiting (needs {MIN_REAL_MONTHS} consecutive months)",
        })
    return pd.DataFrame(rows)


def known_names():
    """The three things every upload (CSV or single submission) is checked against: the list of valid
    district names, the list of valid commodity names, and each commodity's usual (median) price."""
    frames = [load_generated()[REQUIRED_COLUMNS]]
    up = load_uploaded()
    if not up.empty:
        frames.append(up[REQUIRED_COLUMNS])
    allrows = pd.concat(frames, ignore_index=True)
    medians = allrows.groupby("commodity")["price_per_kg"].median()
    return sorted(allrows["location"].unique()), sorted(allrows["commodity"].unique()), medians


def commodity_categories():
    """Maps every commodity name to its category: Chicken, Rice or Fish.

    The category was attached exactly once, in notebook 01, from which raw file each row originally came
    from (chicken.csv / rice.csv / fish.csv) - it is never guessed or re-derived from the commodity's name,
    so there is nothing here that could misclassify a new or oddly-named commodity.
    """
    df = load_district_prices()
    return df.drop_duplicates("commodity").set_index("commodity")["category"].to_dict()


# ================================================================ STATISTICS (admin panel)
def commodity_stats(monthly=None):
    """One row per commodity: its category, how many districts have a price for it, the real 2025 vs 2026
    yearly average and the % change between them, and the latest available monthly price (averaged across
    districts). Used by the Admin > Statistics tab - every number here is a straightforward summary of data
    already loaded elsewhere, nothing is computed specially just for this view.
    """
    district_df = load_district_prices()
    # pivot_table turns the long (Year, commodity, price) table into a wide one with a column per year -
    # this makes "average price in 2025" and "average price in 2026" sit side by side for every commodity.
    pivot = district_df.pivot_table(index=["commodity", "category"], columns="Year",
                                    values="price_per_kg", aggfunc="mean")
    pivot.columns = [f"avg_{int(c)}" for c in pivot.columns]        # "2025" (a Year) -> "avg_2025" (a column name)
    pivot = pivot.reset_index()
    if "avg_2025" in pivot.columns and "avg_2026" in pivot.columns:
        pivot["pct_change_2025_2026"] = (pivot["avg_2026"] / pivot["avg_2025"] - 1) * 100

    districts_covered = district_df.groupby("commodity")["location"].nunique().rename("districts_covered")
    pivot = pivot.merge(districts_covered, on="commodity", how="left")

    monthly = load_effective_monthly() if monthly is None else monthly
    latest_month = monthly["date"].max()
    latest = (monthly[monthly["date"] == latest_month].groupby("commodity")["price_per_kg"]
             .mean().round(2).rename("latest_month_avg_price"))
    pivot = pivot.merge(latest, on="commodity", how="left")

    return pivot.sort_values("commodity").reset_index(drop=True)


def category_summary(stats=None):
    """Rolls commodity_stats() up to one row per category (Chicken / Rice / Fish) - the "By category"
    cards at the top of the Admin > Statistics tab."""
    stats = commodity_stats() if stats is None else stats
    out = stats.groupby("category", as_index=False).agg(
        commodities=("commodity", "count"),
        avg_price_2026=("avg_2026", "mean"),
        avg_pct_change_2025_2026=("pct_change_2025_2026", "mean"),
    )
    return out.round(2)


# ================================================================ COMPARING DISTRICTS (Forecast tab chart)
def commodity_price_by_district(monthly, commodity):
    """The latest available monthly price of one commodity in every district that has it, sorted highest
    first. Used by the Forecast tab's "every district compared" chart, so the district currently being
    forecast can be seen in context next to every other district."""
    sub = monthly[monthly["commodity"] == commodity]
    latest_month = sub["date"].max()
    sub = sub[sub["date"] == latest_month]
    return (sub[["location", "price_per_kg"]]
           .sort_values("price_per_kg", ascending=False).reset_index(drop=True))


# ================================================================ CHECKING AN UPLOADED FILE
def _suggest(name, known):
    """Turns a misspelled name into a "(did you mean 'X'?)" hint, or an empty string if nothing is close."""
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)   # a fuzzy match on spelling (e.g. "Dhak" -> "Dhaka")
    if not close:
        close = [k for k in known if name.lower() in k.lower()][:1]   # a substring match (e.g. "Hilsha" -> "Hilsha (500-900) gm")
    return f" (did you mean '{close[0]}'?)" if close else ""


def validate_upload(raw, known_locations, known_commodities, medians, allow_new=False):
    """Checks every row of an uploaded table (a CSV file, or a single public submission wrapped in a
    one-row table) and splits it into rows that pass and rows that don't.

    Returns a tuple (accepted_rows, rejected_rows, number_of_merged_rows, error_message). `error_message`
    is only set for a problem with the WHOLE file (too many rows, missing columns) - for a per-row problem,
    that row simply ends up in `rejected_rows` with its own reason attached.
    """
    if len(raw) > MAX_UPLOAD_ROWS:
        return None, None, 0, f"The file has more than {MAX_UPLOAD_ROWS:,} rows. Please split it into smaller files."

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]     # accept "Date", " date ", "DATE" etc. as "date"
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, None, 0, ("Missing column(s): " + ", ".join(missing) +
                               ". The file must have exactly these columns: " + ", ".join(REQUIRED_COLUMNS))

    df = df[REQUIRED_COLUMNS].copy()
    df.insert(0, "file_row", range(2, len(df) + 2))         # row 1 of the file is the header, so data starts at row 2
    for c in REQUIRED_COLUMNS:
        df[c] = df[c].fillna("").astype(str).str.strip()     # treat a blank cell the same whether it's empty text or a true NaN
    df = df[~(df[REQUIRED_COLUMNS] == "").all(axis=1)]        # drop completely blank lines (e.g. a trailing empty row)

    # Fix upper/lower case of names so "dhaka" and "Dhaka" are treated as the exact same district.
    loc_map = {n.lower(): n for n in known_locations}
    com_map = {n.lower(): n for n in known_commodities}
    df["location"] = df["location"].map(lambda x: loc_map.get(x.lower(), x))
    df["commodity"] = df["commodity"].map(lambda x: com_map.get(x.lower(), x))

    # Dates: only YYYY-MM-DD or YYYY-MM are accepted. Anything else (like 8/1/2026) is rejected rather than
    # guessed at, because day and month can be swapped between date conventions and guessing wrong would
    # silently corrupt the data - errors="coerce" turns anything that doesn't match into NaT (a missing
    # date) instead of crashing, so it can be caught and reported as a normal rejection below.
    date = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    date = date.fillna(pd.to_datetime(df["date"], format="%Y-%m", errors="coerce"))
    price = pd.to_numeric(df["price_per_kg"].str.replace(",", "", regex=False), errors="coerce")   # "1,013" -> 1013

    reason = pd.Series("", index=df.index, dtype=object)     # "" means "no problem found yet" for that row

    def flag(mask, text):
        # Only the FIRST problem found for each row is kept (a row that is both empty AND has a bad date
        # should show one clear reason, not a confusing list) - `reason == ""` skips rows already flagged.
        mask = mask & (reason == "")
        reason[mask] = text[mask] if isinstance(text, pd.Series) else text

    flag(df["location"] == "", "district is empty")
    flag(df["commodity"] == "", "commodity is empty")
    flag(date.isna(), "date must look like 2026-08-01 (YYYY-MM-DD) or 2026-08 (YYYY-MM)")
    flag(date > current_month_start(), "date is in the future")
    flag(price.isna() | (price <= 0), "price must be a number above 0")
    if not allow_new:
        flag(~df["location"].isin(known_locations),
             df["location"].map(lambda x: f"unknown district '{x}'" + _suggest(x, known_locations)))
        flag(~df["commodity"].isin(known_commodities),
             df["commodity"].map(lambda x: f"unknown commodity '{x}'" + _suggest(x, known_commodities)))
    # This is the COARSE sanity check mentioned in the RISE_ACCEPT_LIMIT comment above: a price more than
    # MAX_PRICE_FACTOR times the commodity's overall median, in EITHER direction, is almost always a typo
    # (a misplaced decimal point, an extra digit) rather than a real price, so it is rejected outright here
    # rather than being sent for the finer per-series review that assess_submission does further down.
    usual = df["commodity"].map(medians)
    flag((price > usual * MAX_PRICE_FACTOR) | (price < usual / MAX_PRICE_FACTOR),
         usual.map(lambda m: f"price is more than {MAX_PRICE_FACTOR}x away from the usual price (about {m:.0f} Tk/kg)"))

    ok = reason == ""
    accepted = pd.DataFrame({
        "date": date[ok].dt.to_period("M").dt.to_timestamp(),   # always store the first day of the month, e.g. 2026-08-01
        "location": df.loc[ok, "location"],
        "commodity": df.loc[ok, "commodity"],
        "price_per_kg": price[ok],
    })
    before = len(accepted)
    # Two rows for the same date+district+commodity (e.g. two different markets reporting the same month)
    # are averaged into one, since the app only ever stores one price per district per commodity per month.
    accepted = accepted.groupby(["date", "location", "commodity"], as_index=False)["price_per_kg"].mean()
    accepted["price_per_kg"] = accepted["price_per_kg"].round(2)
    merged = before - len(accepted)

    rejected = df.loc[~ok, ["file_row"] + REQUIRED_COLUMNS].copy()
    rejected["reason"] = reason[~ok]
    return accepted, rejected, merged, None


# ================================================================ SAVING, BACKUP, UNDO
def save_upload(accepted):
    """Adds already-validated rows to the real uploaded database, backing up the old version first.

    Returns (added, updated): how many rows were brand new versus how many replaced an existing value for
    the same date+district+commodity (the newer value always wins - see drop_duplicates below).
    """
    os.makedirs(BACKUP_DIR, exist_ok=True)
    old = load_uploaded()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    old.to_csv(os.path.join(BACKUP_DIR, f"uploaded_monthly_{stamp}.csv"), index=False)   # snapshot of the state BEFORE this change

    new = accepted.copy()
    new["uploaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    combined = new if old.empty else pd.concat([old, new], ignore_index=True)
    # keep="last" means that when the same date+district+commodity appears twice (an old value and this
    # new one), the NEW row is the one that survives - so re-uploading a correction always wins.
    combined = combined.drop_duplicates(subset=["date", "location", "commodity"], keep="last")
    combined = combined.sort_values(["location", "commodity", "date"]).reset_index(drop=True)
    combined.to_csv(UPLOADED_PATH, index=False)

    updated = len(old) + len(new) - len(combined)     # how many of the new rows overwrote something that was already there
    return len(new) - updated, updated


def undo_last_upload():
    """Restores the uploaded database to whatever it was immediately before the most recent save_upload()
    call, by copying the newest backup file back into place. Returns False if there is nothing to undo."""
    if not os.path.isdir(BACKUP_DIR):
        return False
    backups = sorted(f for f in os.listdir(BACKUP_DIR) if f.startswith("uploaded_monthly_"))   # sorted = oldest to newest, since the filename has a timestamp
    if not backups:
        return False
    last = os.path.join(BACKUP_DIR, backups[-1])
    shutil.copy(last, UPLOADED_PATH)
    os.remove(last)          # consumed - a second "Undo" click goes one step further back, it doesn't repeat this one
    return True


# ================================================================ PUBLIC PRICE SUBMISSIONS
# A member of the public can submit one (date, district, commodity, price) at a time from the "Submit a
# price" tab. Every submission is ALWAYS checked with the same rules as an admin CSV upload first
# (validate_upload, above). What happens next depends on assess_submission (right below): a price that
# looks consistent with that specific series' own recent price is added immediately and the point is
# credited straight away; a price that looks unusual is instead held here with status "pending" until an
# administrator reviews it, and the point only arrives if they approve it.
def load_community_submissions():
    """Every public submission ever made, including auto-approved, pending, approved and rejected ones."""
    if not os.path.exists(COMMUNITY_PATH):
        return pd.DataFrame(columns=COMMUNITY_COLUMNS)
    df = pd.read_csv(COMMUNITY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def assess_submission(monthly, location, commodity, price):
    """Decides whether one candidate price needs an administrator, by comparing it to that EXACT district
    and commodity's own last known price (not the commodity's overall median - that coarser check already
    happened in validate_upload).

    Returns a dict:
      unusual        bool  - True means this needs an administrator before it becomes real data
      reference_price      - the last known price it was compared against, or None if this series has no
                              earlier data at all to compare against
      pct_change            - the % change versus reference_price, signed (positive = a rise)
      severity        "moderate" | "severe" | None  - only set when unusual is True; "severe" is a plain
                              label for how big the move is, it does not change what happens next (both
                              moderate and severe are held for review either way)
      reason                - a plain-language explanation, only set when unusual is True

    A series with no prior data at all is always treated as unusual: there is nothing to measure a "normal"
    move against yet, so an administrator has to be the one to confirm the very first price for it.
    """
    sub = monthly[(monthly["location"] == location) & (monthly["commodity"] == commodity)]
    if sub.empty:
        return {"unusual": True, "reference_price": None, "pct_change": None, "severity": "moderate",
               "reason": "there is no earlier price for this district and commodity to compare it to"}

    last_price = float(sub.sort_values("date")["price_per_kg"].iloc[-1])     # the most recent month on record for this exact series
    pct_change = (price - last_price) / last_price * 100                     # e.g. +30 means "30% higher than last time"

    # A rise and a fall are judged against different limits (see the RISE_ACCEPT_LIMIT comment near the top
    # of this file for why) - pick the right pair of limits for the direction this particular price moved.
    if pct_change >= 0:
        accept_limit, severe_limit, direction = RISE_ACCEPT_LIMIT, RISE_SEVERE_LIMIT, "rise"
    else:
        accept_limit, severe_limit, direction = FALL_ACCEPT_LIMIT, FALL_SEVERE_LIMIT, "drop"

    unusual = abs(pct_change) > accept_limit * 100
    reason = severity = None
    if unusual:
        severity = "severe" if abs(pct_change) >= severe_limit * 100 else "moderate"
        reason = f"a {abs(pct_change):.0f}% {direction} from the last known price ({last_price:.2f} Tk/kg)"
    return {"unusual": unusual, "reference_price": last_price, "pct_change": pct_change,
           "severity": severity, "reason": reason}


def submit_community_price(date, location, commodity, price_per_kg, contributor, monthly,
                           known_locations, known_commodities, medians):
    """Validates one public submission, then either adds it immediately or holds it for an administrator.

    `contributor` is the logged-in name/phone number - not a verified identity, just the account the point
    gets credited to. Returns a tuple (accepted, status, info):
      accepted=False                     - rejected by validate_upload's normal checks (bad format, unknown
                                            name, future date, price far from the commodity's usual range,
                                            or no contributor logged in); `info` is the reason, as a string.
      accepted=True, status="auto_approved" - the price matched the series' own recent trend closely enough,
                                            so it is already saved and the point has already been credited;
                                            `info` is the assess_submission() dict.
      accepted=True, status="pending"    - the price looked unusual, so it is waiting for an administrator;
                                            `info` is the assess_submission() dict (its "reason" field says why).
    """
    if not str(contributor).strip():
        return False, None, "Please log in with a name or phone number first."

    # Step 1: the same coarse checks an admin CSV upload goes through (wrapped as a one-row table so the
    # exact same validate_upload() function can be reused unchanged).
    raw = pd.DataFrame([{"date": date, "location": location, "commodity": commodity, "price_per_kg": price_per_kg}])
    accepted, rejected, merged, error = validate_upload(raw, known_locations, known_commodities, medians)
    if error:
        return False, None, error
    if len(rejected) > 0:
        return False, None, rejected.iloc[0]["reason"]

    # Step 2: the finer, series-specific check that decides immediate vs held-for-review.
    row = accepted.iloc[0]
    assessment = assess_submission(monthly, row["location"], row["commodity"], float(row["price_per_kg"]))
    status = "pending" if assessment["unusual"] else "auto_approved"

    # Every submission is logged here regardless of outcome, so there is always a full audit trail of what
    # was submitted, by whom, when, and what was decided (and why, for anything that was flagged).
    entry = pd.DataFrame([{
        "date": row["date"].strftime("%Y-%m-%d"),
        "location": row["location"],
        "commodity": row["commodity"],
        "price_per_kg": row["price_per_kg"],
        "contributor": str(contributor).strip(),
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": status,
        "reference_price": assessment["reference_price"],
        "pct_change": assessment["pct_change"],
        "severity": assessment["severity"],
        "reason": assessment["reason"],
    }])
    os.makedirs(os.path.dirname(COMMUNITY_PATH) or ".", exist_ok=True)
    entry.to_csv(COMMUNITY_PATH, mode="a", header=not os.path.exists(COMMUNITY_PATH), index=False)

    if status == "auto_approved":
        # .to_frame().T turns the single pandas Series `row` back into a one-row DataFrame, which is what
        # save_upload() expects (the same function the Admin CSV-upload path uses for a whole file).
        save_upload(row[["date", "location", "commodity", "price_per_kg"]].to_frame().T)
        award_point(contributor)
    return True, status, assessment


def pending_community_submissions():
    """Rows currently flagged as unusual and waiting on an administrator, with their position in the file
    (row_id) attached so a single one can be approved or rejected. Auto-approved rows never appear here -
    only ones that genuinely need a human decision do."""
    df = load_community_submissions()
    pending = df[df["status"] == "pending"]
    return pending.reset_index().rename(columns={"index": "row_id"})


def review_community_submission(row_id, approve):
    """An administrator's decision on one flagged row.

    Approving it adds it to the real uploaded database via save_upload() - the exact same path and the
    exact same backup-then-write behaviour as a CSV upload - and credits the contributor one point.
    Rejecting it only changes its status to "rejected"; nothing else is touched and no point is given.
    """
    df = load_community_submissions()
    if row_id not in df.index:
        return False
    if approve:
        row = df.loc[[row_id], ["date", "location", "commodity", "price_per_kg"]].copy()
        save_upload(row)
        award_point(df.loc[row_id, "contributor"])
        df.loc[row_id, "status"] = "approved"
    else:
        df.loc[row_id, "status"] = "rejected"
    df.to_csv(COMMUNITY_PATH, index=False)
    return True


def submission_stats():
    """A summary of every public submission ever made, for the Admin > Results tab: how many were added
    immediately, how many are waiting, how many an administrator approved or rejected, and how many of the
    ones that needed a decision were "moderate" versus "severe" (see assess_submission)."""
    df = load_community_submissions()
    statuses = ("auto_approved", "pending", "approved", "rejected")
    if df.empty:
        return {"total": 0, **{s: 0 for s in statuses}, "moderate": 0, "severe": 0}

    by_status = {s: int((df["status"] == s).sum()) for s in statuses}
    flagged = df[df["status"].isin(["pending", "approved", "rejected"])]     # every submission that was NOT auto-approved
    return {
        "total": len(df),
        **by_status,
        "moderate": int((flagged["severity"] == "moderate").sum()),
        "severe": int((flagged["severity"] == "severe").sum()),
    }


# ================================================================ POINTS LEDGER
def load_points_ledger():
    """Every points event ever logged: who, what kind of event, how many points, and when."""
    if not os.path.exists(POINTS_PATH):
        return pd.DataFrame(columns=POINTS_COLUMNS)
    return pd.read_csv(POINTS_PATH)


def _log_points_event(contributor, event, points):
    """Appends one row to the points ledger. Never edits or removes a past row - a contributor's balance
    is always just the sum of every event they have ever had (see contributor_status below), so the ledger
    doubles as a full history of how they earned or spent every point."""
    entry = pd.DataFrame([{"contributor": str(contributor).strip(), "event": event, "points": points,
                          "time": datetime.now().strftime("%Y-%m-%d %H:%M")}])
    os.makedirs(os.path.dirname(POINTS_PATH) or ".", exist_ok=True)
    entry.to_csv(POINTS_PATH, mode="a", header=not os.path.exists(POINTS_PATH), index=False)


def award_point(contributor):
    """+1 point, logged as earned from a submission (whether it was auto-approved or an admin approved it)."""
    _log_points_event(contributor, "earned_submission", 1)


def contributor_status(contributor):
    """How many free Budget planner runs and how many points a contributor currently has left."""
    ledger = load_points_ledger()
    mine = ledger[ledger["contributor"] == str(contributor).strip()]
    free_used = int((mine["event"] == "free_trial_used").sum())
    points_balance = int(mine["points"].sum()) if not mine.empty else 0    # "spent_point" events are logged as -1, so a simple sum nets out correctly
    return {"free_remaining": max(FREE_BUDGET_PLANS - free_used, 0), "points_balance": points_balance}


def consume_budget_plan_credit(contributor):
    """Call this right before actually running the Budget planner. Uses up a free trial first if any are
    left, otherwise spends one point. Raises ValueError (with a message that is safe to show the person
    directly) if neither is available."""
    if not str(contributor).strip():
        raise ValueError("Please enter your name or phone number first.")
    status = contributor_status(contributor)
    if status["free_remaining"] > 0:
        _log_points_event(contributor, "free_trial_used", 0)      # 0 points - this event only exists to be counted against FREE_BUDGET_PLANS
        return "free_trial"
    if status["points_balance"] > 0:
        _log_points_event(contributor, "spent_point", -1)
        return "point"
    raise ValueError(f"No free trials or points left for '{contributor}'. Submit a commodity price and have "
                     "an administrator approve it to earn another point.")


def points_leaderboard(top_n=10):
    """The contributors with the most points right now, for the Admin > Results tab. This is purely an
    informational view - it has no effect on how points are earned or spent."""
    ledger = load_points_ledger()
    if ledger.empty:
        return pd.DataFrame(columns=["contributor", "points_balance"])
    board = ledger.groupby("contributor")["points"].sum().reset_index().rename(columns={"points": "points_balance"})
    board = board[board["points_balance"] > 0]         # someone who has only ever used free trials has 0 points - not a meaningful "leader"
    return board.sort_values("points_balance", ascending=False).head(top_n).reset_index(drop=True)


# ================================================================ TRAINING THE FORECASTING MODEL
# ---- How the model "thinks" -------------------------------------------------------------------
# The model's job is: given everything about ONE month for ONE district+commodity, predict that same
# series' price the FOLLOWING month. It is not shown "the future" in any other sense - it only ever learns
# the relationship between one month's numbers and the very next month's price, and forecasting several
# months ahead (see forecast_series, further down) works by repeatedly feeding its own predictions back in
# as if they were the "current month", one step at a time.
#
# The six numbers ("features") the model is shown for one month, and why each one is there:
#   price_per_kg              this month's price - the single strongest clue for next month's price
#   lag1, lag2, lag3          the price 1, 2 and 3 months before this one - lets the model notice a trend
#                             (steadily rising/falling) rather than reacting to one month in isolation
#   month_num                 the calendar month (1-12) - lets the model learn a seasonal pattern if one
#                             exists (e.g. a crop that is always cheaper right after its harvest month)
#   time_index                a plain incrementing count of months since Jan 2025 - lets the model represent
#                             an overall long-run drift across the two years of data, separately from season
#   location_<District>       one column per district, holding 1 for the matching district and 0 for all
#   commodity_<Commodity>     others - this "one-hot" encoding lets a SINGLE model be shared across every
#                             district and commodity, rather than training hundreds of separate tiny models
#
# The model itself is XGBoost: a "gradient-boosted trees" model. In plain terms, it is a sequence of many
# small decision trees (n_estimators=200 of them below), where each new tree is trained specifically to
# correct the mistakes the trees before it were still making, and the final prediction is the combined
# result of all of them together. It is a strong, fast, and easy-to-explain default choice for exactly this
# kind of "structured numbers in a table" problem - and unlike some other tree-based methods it comes with
# very fast prediction, which matters here because forecast_series calls the model once per future month.


def build_training_table(monthly):
    """Turns the plain monthly price history into the table shape a model can actually learn from: one row
    per (district, commodity, month), with that month's price, its 3 preceding lags, the month number, the
    time index, and - critically - the price of the FOLLOWING month attached as "target_next_month" (what
    the model is trying to learn to predict). Rows where a lag or the target does not exist yet (the very
    first few months of a series, or its very last month, which has no "next month" to learn from) are
    dropped by dropna() at the end, since they cannot be used for training either way.
    """
    df = monthly[REQUIRED_COLUMNS + ["source"]].sort_values(["location", "commodity", "date"]).reset_index(drop=True)
    grouped = df.groupby(["location", "commodity"])["price_per_kg"]     # each district+commodity's price history, kept separate
    df["month_num"] = df["date"].dt.month
    df["time_index"] = (df["date"].dt.year - 2025) * 12 + df["date"].dt.month - 1     # 0 for Jan 2025, 1 for Feb 2025, ... counting up
    for lag in (1, 2, 3):
        # .shift(lag) moves each series' own price history down by `lag` rows, so lag1 on the March row
        # holds February's price, lag2 holds January's, and so on - shift() never mixes one district's or
        # commodity's prices into another's, because it operates on each `grouped` series separately.
        df[f"lag{lag}"] = grouped.shift(lag)
    df["target_next_month"] = grouped.shift(-1)      # shift(-1): the row BELOW this one - i.e. next month's price
    return df.dropna().reset_index(drop=True)


def _new_model():
    """Creates a fresh, untrained XGBoost model with the settings used everywhere in this file.

    n_estimators=200    how many small decision trees are combined together (more trees can capture more
                         detail, but also risk overfitting and take longer to train/predict)
    max_depth=4          how many yes/no questions deep each individual tree is allowed to go - kept
                         shallow on purpose, since a market-price dataset this size does not have enough
                         signal to justify deep, highly specific trees (they would just memorise noise)
    learning_rate=0.05    how much each new tree is allowed to correct the ones before it - a smaller value
                         means slower, steadier learning, which tends to generalise better than large,
                         aggressive corrections
    random_state=42      fixes the "random" parts of training (how the trees are built) to a specific seed,
                         so training on the exact same data twice always produces the exact same model -
                         useful for reproducibility and for the tests in tests/test_project.py
    """
    return xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)


def save_model(model, features, model_path=MODEL_PATH, features_path=FEATURES_PATH):
    """Writes a trained model and its exact list of expected input columns to disk."""
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save_model(model_path)                                  # XGBoost's own .save_model() writes the JSON format described near MODEL_PATH above
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(list(features), f, indent=1)                     # the column names AND their order both matter - see load_model below


def load_model(model_path=MODEL_PATH, features_path=FEATURES_PATH):
    """Loads a trained model and its feature list back from disk. Returns (model, feature_columns).
    Raises ValueError with a message that is safe to show directly in the app if a file is missing,
    unreadable, or does not match its own feature list (which would mean the two files got out of sync -
    for example if only one of the pair was copied somewhere)."""
    for path in (model_path, features_path):
        if not os.path.exists(path):
            raise ValueError(f"'{path}' is missing. Run notebooks/02_train_forecast_model.ipynb to create it.")
    try:
        model = _new_model()                 # XGBoost's load_model fills in an existing model object - it needs one with the same settings first
        model.load_model(model_path)
        with open(features_path, encoding="utf-8") as f:
            features = json.load(f)
    except Exception as error:
        raise ValueError(f"The model files could not be read ({type(error).__name__}). They may have been damaged "
                         "while copying. Run notebooks/02_train_forecast_model.ipynb to rebuild them.") from error
    if model.n_features_in_ != len(features):
        raise ValueError("The model and its list of input columns do not match. "
                         "Run notebooks/02_train_forecast_model.ipynb to rebuild them.")
    return model, features


def train_candidate(monthly, test_months=3):
    """Trains a brand NEW model from scratch and honestly tests how good it is, without touching the model
    that is currently live. This is what powers the Admin > Retrain model tab's "Train (test only)" button -
    nothing is replaced until a second, separate step (install_model) is confirmed.

    The testing method (why it is done this way, not just "check accuracy on the training data"):
      1. The most recent `test_months` months of every series are set aside and NOT shown to the model
         during training at all - they are the "exam questions".
      2. A model is trained only on everything OLDER than that.
      3. That model then predicts the set-aside months, and its predictions are compared to what the price
         actually was - this is the only honest way to estimate how the model will do on a month it has
         genuinely never seen, because testing it on data it was trained on would just measure how well it
         memorised the past, not how well it predicts the future.
      4. Its error (MAE, see below) is compared against the simplest possible forecast - "next month's price
         will be the same as this month's" - as a sanity-check baseline. A model that cannot even beat that
         trivial guess is not worth using.
      5. Only AFTER that honest test is complete does a second, final model get trained on every month
         available (including the test months) - that last model is the one actually offered for saving,
         since by then it has already proven itself and there is no reason to withhold the most recent data
         from it.
    """
    data = monthly[monthly["date"] <= current_month_start()]          # never train (or test) on a future-dated month
    table = build_training_table(data)
    months = sorted(table["date"].drop_duplicates())
    if len(table) < 200 or len(months) < test_months + 3:
        raise ValueError("Not enough data to train and test a model.")

    is_test = table["date"].isin(months[-test_months:])               # True for rows in the held-out "exam" months
    # pd.get_dummies turns the single "location" and "commodity" text columns into many 0/1 columns (one
    # per district, one per commodity) - this is the "one-hot encoding" mentioned above; XGBoost (like most
    # ML models) needs numbers, not text, as its input.
    enc = pd.get_dummies(table, columns=["location", "commodity"])
    feature_cols = [c for c in enc.columns if c not in ("date", "target_next_month", "source")]
    X, y = enc[feature_cols], enc["target_next_month"]                 # X = the inputs, y = the "answer" the model is trying to learn

    test_model = _new_model().fit(X[~is_test], y[~is_test])            # step 1-2: learn ONLY from the months before the test window
    pred = test_model.predict(X[is_test])                              # step 3: predict the held-out months
    y_test = y[is_test]
    mae_model = float((y_test - pred).abs().mean())                    # Mean Absolute Error: the average size of the model's mistake, in Tk/kg
    mae_baseline = float((y_test - X.loc[is_test, "price_per_kg"]).abs().mean())   # step 4: baseline = "next month = this month"
    mape_model = float(((y_test - pred).abs() / y_test).mean() * 100)  # the same error expressed as a %, easier to compare across cheap vs expensive commodities

    final_model = _new_model().fit(X, y)                               # step 5: the model actually offered for saving, trained on everything
    return {
        "model": final_model,
        "features": feature_cols,
        "metrics": {
            "rows": int(len(table)),
            "series": int(table.groupby(["location", "commodity"]).ngroups),
            "test_months": test_months,
            "test_rows": int(is_test.sum()),
            "real_test_rows": int((is_test & (table["source"] == "uploaded")).sum()),    # how many of the "exam" rows were genuine, not generated, data
            "mae_model": mae_model,
            "mae_baseline": mae_baseline,
            "mape_model": mape_model,
            "passed": mae_model <= mae_baseline,                       # did the model actually beat the trivial baseline?
        },
    }


def evaluate_current_model(monthly, model, features, test_months=3):
    """Tests the model that is CURRENTLY installed and live in the Forecast tab right now, using the exact
    same honest hold-out method as train_candidate() above - but WITHOUT training anything new. This
    answers a different question than train_candidate does: train_candidate asks "would a freshly retrained
    model do better than the baseline?", while this asks "how accurate is the model that is actually running
    right now?" - which is what the Admin > Results tab reports as the app's current accuracy.
    """
    data = monthly[monthly["date"] <= current_month_start()]
    table = build_training_table(data)
    months = sorted(table["date"].drop_duplicates())
    if len(table) < 50 or len(months) < test_months + 1:
        raise ValueError("Not enough data yet to evaluate the current model.")

    is_test = table["date"].isin(months[-test_months:])
    enc = pd.get_dummies(table, columns=["location", "commodity"])
    # The held-out months might not happen to include every single district or commodity, so pd.get_dummies
    # on just this slice could be missing a location_/commodity_ column the model still expects as an input.
    # Add any missing one back in as all-zeros, so the column set always exactly matches what the model was
    # trained on - a plain `enc[features]` below would otherwise raise a KeyError for a missing column.
    for col in features:
        if col not in enc.columns:
            enc[col] = 0

    X, y = enc[features], enc["target_next_month"]
    pred = model.predict(X[is_test])                     # using the LIVE model passed in - nothing is trained here
    y_test = y[is_test]
    mae_model = float((y_test - pred).abs().mean())
    mae_baseline = float((y_test - X.loc[is_test, "price_per_kg"]).abs().mean())
    mape_model = float(((y_test - pred).abs() / y_test).mean() * 100)
    return {
        "test_months": test_months,
        "test_rows": int(is_test.sum()),
        "real_test_rows": int((is_test & (table["source"] == "uploaded")).sum()),
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "mape_model": mape_model,
        "beats_baseline": mae_model <= mae_baseline,
    }


def install_model(candidate):
    """Makes a trained candidate (from train_candidate) the live model: backs up whatever model is
    currently installed, saves the new one in its place, and appends one row to the training log so
    accuracy over successive retrains can be tracked (see the Admin > Results tab)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(MODEL_BACKUP_DIR, stamp)
    os.makedirs(folder, exist_ok=True)
    for path in (MODEL_PATH, FEATURES_PATH):
        if os.path.exists(path):
            shutil.copy(path, folder)              # back up BEFORE overwriting, same safety pattern as save_upload()
    save_model(candidate["model"], candidate["features"])

    m = candidate["metrics"]
    entry = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows": m["rows"], "series": m["series"], "test_rows": m["test_rows"],
        "mae_model": round(m["mae_model"], 2), "mae_baseline": round(m["mae_baseline"], 2),
    }])
    if os.path.exists(TRAINING_LOG):
        entry = pd.concat([pd.read_csv(TRAINING_LOG), entry], ignore_index=True)     # append to the history, don't overwrite it
    entry.to_csv(TRAINING_LOG, index=False)


# ================================================================ DISTRICT PRICES (Cross-district tab)
def load_district_prices():
    """The real yearly average price per district and commodity, straight from the DAM files (via
    notebook 01) - this is what the Cross-district tab's price-ratio estimate is built from."""
    df = pd.read_csv(DISTRICT_PATH)
    df["Year"] = df["Year"].astype(int)
    return df


def common_commodities(district_df, district_a, district_b):
    """Which commodities have a real price recorded in BOTH given districts (the set intersection `&`) -
    the Cross-district tab can only compare a commodity the two districts actually share."""
    a = set(district_df.loc[district_df["location"] == district_a, "commodity"])
    b = set(district_df.loc[district_df["location"] == district_b, "commodity"])
    return sorted(a & b)


def cross_district_estimate(district_df, source, target, commodity):
    """Price-ratio method: roughly how much more (or less) expensive is the target district than the
    source district, for one commodity?

    For every year where BOTH districts have a recorded price, the target's price is divided by the
    source's price for that year, giving that year's ratio; the average of all such yearly ratios is the
    single number used to convert a price in the source district into an estimate for the target district.
    """
    if source == target:
        raise ValueError("Please choose two different districts.")
    sub = district_df[(district_df["commodity"] == commodity) & (district_df["location"].isin([source, target]))]
    table = sub.pivot_table(index="location", columns="Year", values="price_per_kg")
    table = table.dropna(axis=1)          # keep only the years where BOTH districts actually have a price
    if table.shape[1] == 0 or source not in table.index or target not in table.index:
        raise ValueError("No year has a price in both districts for this commodity.")
    yearly_ratio = table.loc[target] / table.loc[source]
    return {
        "table": table,                                   # the raw year-by-district price table (shown in the admin-only detail view)
        "yearly_ratio": yearly_ratio,
        "ratio": float(yearly_ratio.mean()),               # the single number actually used for the estimate
        "typical_source": float(table.loc[source].mean()),
        "typical_target": float(table.loc[target].mean()),
    }


# ================================================================ FORECASTING (Forecast tab)
def make_feature_row(features, prices, last_date, last_time_index, step, location, commodity):
    """Builds ONE row of model input, describing the month the forecast is currently "standing in" - the
    model then predicts the price of the month right after this one.

    `prices` is the running list of known-plus-already-forecast prices for this series (see forecast_series
    below, which appends each new prediction onto the end of this same list as it goes) - so prices[-1] is
    always "this month's price" from the model's point of view, prices[-2] is one month before that, and
    so on, exactly matching how lag1/lag2/lag3 were built during training in build_training_table.
    """
    row = {col: 0 for col in features}          # start every one-hot location_/commodity_ column at 0
    row["price_per_kg"] = prices[-1]
    row["lag1"] = prices[-2]
    row["lag2"] = prices[-3]
    row["lag3"] = prices[-4]
    row["time_index"] = last_time_index + step
    row["month_num"] = (last_date + pd.DateOffset(months=step)).month
    for key in (f"location_{location}", f"commodity_{commodity}"):
        if key in row:                          # switch on just the ONE district column and ONE commodity column that apply here
            row[key] = 1
    return row


def forecast_series(monthly, model, features, location, commodity, months_ahead, today=None, history_months=12):
    """Forecasts one district+commodity's price, `months_ahead` months into the future, always starting
    from the month right after `today` (or right after the real "today" if not given - `today` mainly
    exists so the tests in tests/test_project.py can check specific dates without waiting for the calendar).

    HOW MULTI-MONTH FORECASTING WORKS: the model only ever knows how to predict ONE month ahead. To get
    several months out, this function calls the model once per future month, and each time, treats its own
    most recent prediction as if it were a real known price for the next call - "recursively" feeding
    predictions back in as input. This is also exactly why forecasts get less trustworthy the further out
    they go (and why the app's disclaimers say so): month 6's forecast is not built from 6 months of real
    data, it is built from 5 of the model's OWN earlier guesses plus 1 real month, so any small error made
    early on can compound by the time it reaches the far end of the forecast.
    """
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    current_month = pd.Timestamp(today.year, today.month, 1)

    hist = monthly[(monthly["location"] == location) & (monthly["commodity"] == commodity)
                   & (monthly["date"] <= current_month)].sort_values("date")
    if len(hist) < 4:
        raise ValueError("Not enough monthly history for this district and commodity.")

    prices = hist["price_per_kg"].tolist()          # this list grows by one every loop iteration below - see make_feature_row's docstring
    last_date = hist["date"].max()
    last_time_index = (last_date.year - 2025) * 12 + (last_date.month - 1)      # months since Jan 2025, matching build_training_table
    # If the data's last known month is already behind today (e.g. this month's price has not been added
    # yet), `skip` counts how many months need to be silently predicted just to catch up to today, BEFORE
    # the actually-requested forecast months begin - those "catching up" months are computed (the model
    # still needs them as stepping stones) but never shown to the person, since they asked for the future,
    # not for months that have already happened.
    skip = (current_month.year - last_date.year) * 12 + (current_month.month - last_date.month)

    rows = []
    for step in range(skip + months_ahead):
        row = make_feature_row(features, prices, last_date, last_time_index, step, location, commodity)
        price = float(model.predict(pd.DataFrame([row])[features])[0])     # [features] enforces the exact column order the model expects
        if step >= skip:                                                    # only keep months from "next month" onwards (hide the catch-up months)
            rows.append({"date": last_date + pd.DateOffset(months=step + 1), "price": price})
        prices.append(price)          # feed this prediction back in, so the NEXT loop iteration's lag1 is THIS iteration's predicted price

    return {
        "forecast": pd.DataFrame(rows),
        "history": hist.tail(history_months)[["date", "price_per_kg"]].reset_index(drop=True),   # just the recent past, for the chart
        "last_known": last_date,
        "source": str(hist["source"].iloc[-1]) if "source" in hist.columns else "generated",       # is the forecast built on real or generated data?
    }


# ================================================================ BUDGET PLANNER
def budget_opportunities(monthly, model, features, budget, months_ahead, location=None,
                         exclude_categories=("Fish",)):
    """Ranks district/commodity combinations by predicted price movement over `months_ahead`, as a rough
    "what might be worth buying and holding" guide.

    The idea: a commodity whose price is predicted to rise faster than others might be worth buying now
    rather than later - so this calls forecast_series() for every eligible combination, ranks the results
    by predicted % change, keeps only the ones the given budget can currently afford at least 1 kg of, and
    (by default) drops any category in `exclude_categories` - fresh fish does not store the way rice or
    other durable goods do, so "buy now, use later" does not make sense for it the way it might for rice.

    This reuses forecast_series() - there is no separate ranking model, it is just the same one-month-ahead
    model called many times. The ranking is therefore only ever as reliable as an individual forecast is
    (see forecast_series' own docstring): on a series still using generated monthly data, an apparent "fast
    riser" often just reflects the straight-line gap between the 2025 and 2026 yearly averages, not a real
    observed trend.
    """
    if budget <= 0:
        raise ValueError("Budget must be greater than 0.")
    cat_map = commodity_categories()
    combos = monthly[["location", "commodity"]].drop_duplicates()
    if location is not None:
        combos = combos[combos["location"] == location]
    combos = combos[~combos["commodity"].map(cat_map).isin(exclude_categories)]

    rows = []
    for loc, com in combos.itertuples(index=False):
        try:
            result = forecast_series(monthly, model, features, loc, com, months_ahead)
        except ValueError:
            continue        # not enough history for this one combination - skip it, don't fail the whole ranking over it
        current_price = float(result["history"]["price_per_kg"].iloc[-1])
        future_price = float(result["forecast"]["price"].iloc[-1])
        if current_price <= 0 or current_price > budget:
            continue        # the budget cannot afford even 1 kg of this today
        rows.append({
            "location": loc,
            "commodity": com,
            "category": cat_map.get(com, "Unknown"),
            "current_price": round(current_price, 2),
            "predicted_price": round(future_price, 2),
            "predicted_change_pct": round((future_price / current_price - 1) * 100, 2),
            "kg_affordable_now": round(budget / current_price, 2),
            "data_basis": result["source"],
        })
    out = pd.DataFrame(rows, columns=["location", "commodity", "category", "current_price", "predicted_price",
                                      "predicted_change_pct", "kg_affordable_now", "data_basis"])
    return out.sort_values("predicted_change_pct", ascending=False).reset_index(drop=True)


# ================================================================ SMALL HELPERS FOR THE APP
def data_signature():
    """A value that changes whenever any of the price data files change on disk - used as the cache key for
    Streamlit's @st.cache_data, so the app never keeps showing stale data after a file is updated."""
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (GENERATED_PATH, UPLOADED_PATH, DISTRICT_PATH))


def model_signature():
    """Same idea as data_signature(), but for the model files - used as the cache key for the loaded model."""
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (MODEL_PATH, FEATURES_PATH))


def dataset_summary(monthly):
    """A handful of top-line numbers about the current dataset, shown as the metric cards at the top of
    Admin > Status: how many districts and price series exist, the latest month covered, and how many of
    those series are already using real (not generated) data."""
    used = monthly[monthly["date"] <= current_month_start()]
    real = used[used["source"] == "uploaded"] if "source" in used.columns else used.iloc[0:0]
    return {
        "districts": int(used["location"].nunique()),
        "series": int(used.groupby(["location", "commodity"]).ngroups),
        "latest_month": used["date"].max(),
        "real_series": int(real.groupby(["location", "commodity"]).ngroups),
    }
