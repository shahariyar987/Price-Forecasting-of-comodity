"""Data, forecasting and model utilities for the Bangladesh Commodity Price Predictor.

The Streamlit app (app.py) only handles the user interface. Everything that computes something
lives here, so it can be tested (tests/) and reused by the notebooks.
"""
import os
import shutil
import difflib
from datetime import datetime

import json

import pandas as pd
import xgboost as xgb

# ---------------- settings you can change ----------------
GENERATED_PATH = "data/processed/monthly_hybrid.csv"      # the generated monthly data (never modified)
UPLOADED_PATH = "data/processed/uploaded_monthly.csv"     # rows uploaded by the admin
DISTRICT_PATH = "data/processed/district_prices_clean.csv"   # real yearly average prices per district (DAM)
BACKUP_DIR = "data/backups"
# XGBoost's own JSON format is used instead of pickle: it is text, it survives copying and
# unzipping unchanged, and it can be read by other XGBoost versions.
MODEL_PATH = "models/model1_forecast.json"
FEATURES_PATH = "models/model1_features.json"
MODEL_BACKUP_DIR = "models/backups"
TRAINING_LOG = "models/training_log.csv"

REQUIRED_COLUMNS = ["date", "location", "commodity", "price_per_kg"]
MIN_REAL_MONTHS = 6    # consecutive uploaded months needed before a series switches from generated to real data
MAX_PRICE_FACTOR = 3   # a price more than 3x above/below the usual price of the commodity is rejected
MAX_UPLOAD_ROWS = 50000   # safety limit for one uploaded file


def current_month_start():
    today = pd.Timestamp.today()
    return pd.Timestamp(today.year, today.month, 1)


# ---------------- loading the data ----------------
def load_generated():
    df = pd.read_csv(GENERATED_PATH, parse_dates=["date"])
    df["source"] = "generated"
    return df


def load_uploaded():
    if not os.path.exists(UPLOADED_PATH):
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["uploaded_at"])
    df = pd.read_csv(UPLOADED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def uploaded_runs():
    """For every uploaded series (district + commodity): all its rows and its newest run of consecutive months."""
    up = load_uploaded()
    runs = {}
    for (loc, com), g in up.groupby(["location", "commodity"]):
        g = g.sort_values("date")
        month_no = g["date"].dt.year * 12 + g["date"].dt.month
        run_id = (month_no.diff() != 1).cumsum()          # a new run starts wherever a month is missing
        runs[(loc, com)] = (g, g[run_id == run_id.iloc[-1]])
    return runs


def load_effective_monthly():
    """The data the app really uses: generated data, except for series that have enough REAL uploaded months."""
    gen = load_generated()
    cols = REQUIRED_COLUMNS + ["source"]
    real_parts, real_keys = [], []
    for key, (all_rows, run) in uploaded_runs().items():
        if len(run) >= MIN_REAL_MONTHS:
            part = run.copy()
            part["source"] = "uploaded"
            real_parts.append(part[cols])
            real_keys.append(key)
    if not real_keys:
        return gen[cols]
    idx = pd.MultiIndex.from_frame(gen[["location", "commodity"]])
    gen_keep = gen[~idx.isin(real_keys)]
    out = pd.concat([gen_keep[cols]] + real_parts, ignore_index=True)
    return out.sort_values(["location", "commodity", "date"]).reset_index(drop=True)


def uploaded_series_status():
    rows = []
    for (loc, com), (all_rows, run) in uploaded_runs().items():
        ready = len(run) >= MIN_REAL_MONTHS
        rows.append({
            "location": loc,
            "commodity": com,
            "months_uploaded": len(all_rows),
            "consecutive_months": len(run),
            "last_month": run["date"].max().strftime("%Y-%m"),
            "status": "using uploaded data" if ready else f"waiting (needs {MIN_REAL_MONTHS} consecutive months)",
        })
    return pd.DataFrame(rows)


def known_names():
    """District names, commodity names and the usual price of each commodity (used to check uploads)."""
    frames = [load_generated()[REQUIRED_COLUMNS]]
    up = load_uploaded()
    if not up.empty:
        frames.append(up[REQUIRED_COLUMNS])
    allrows = pd.concat(frames, ignore_index=True)
    medians = allrows.groupby("commodity")["price_per_kg"].median()
    return sorted(allrows["location"].unique()), sorted(allrows["commodity"].unique()), medians


# ---------------- checking an uploaded file ----------------
def _suggest(name, known):
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    if not close:
        close = [k for k in known if name.lower() in k.lower()][:1]     # e.g. "Hilsha" -> "Hilsha (500-900) gm"
    return f" (did you mean '{close[0]}'?)" if close else ""


def validate_upload(raw, known_locations, known_commodities, medians, allow_new=False):
    """Returns (accepted_rows, rejected_rows, number_of_merged_rows, error_message)."""
    if len(raw) > MAX_UPLOAD_ROWS:
        return None, None, 0, f"The file has more than {MAX_UPLOAD_ROWS:,} rows. Please split it into smaller files."
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, None, 0, ("Missing column(s): " + ", ".join(missing) +
                               ". The file must have exactly these columns: " + ", ".join(REQUIRED_COLUMNS))

    df = df[REQUIRED_COLUMNS].copy()
    df.insert(0, "file_row", range(2, len(df) + 2))          # row 1 of the file is the header
    for c in REQUIRED_COLUMNS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df = df[~(df[REQUIRED_COLUMNS] == "").all(axis=1)]        # completely blank lines

    # fix upper/lower case of names ("dhaka" -> "Dhaka")
    loc_map = {n.lower(): n for n in known_locations}
    com_map = {n.lower(): n for n in known_commodities}
    df["location"] = df["location"].map(lambda x: loc_map.get(x.lower(), x))
    df["commodity"] = df["commodity"].map(lambda x: com_map.get(x.lower(), x))

    # dates: only year-month-day or year-month (other formats are ambiguous, e.g. 8/1/2026)
    date = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    date = date.fillna(pd.to_datetime(df["date"], format="%Y-%m", errors="coerce"))
    price = pd.to_numeric(df["price_per_kg"].str.replace(",", "", regex=False), errors="coerce")

    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask, text):
        mask = mask & (reason == "")                          # keep only the first problem of each row
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
    usual = df["commodity"].map(medians)
    flag((price > usual * MAX_PRICE_FACTOR) | (price < usual / MAX_PRICE_FACTOR),
         usual.map(lambda m: f"price is more than {MAX_PRICE_FACTOR}x away from the usual price (about {m:.0f} Tk/kg)"))

    ok = reason == ""
    accepted = pd.DataFrame({
        "date": date[ok].dt.to_period("M").dt.to_timestamp(),   # always the first day of the month
        "location": df.loc[ok, "location"],
        "commodity": df.loc[ok, "commodity"],
        "price_per_kg": price[ok],
    })
    before = len(accepted)
    accepted = accepted.groupby(["date", "location", "commodity"], as_index=False)["price_per_kg"].mean()
    accepted["price_per_kg"] = accepted["price_per_kg"].round(2)
    merged = before - len(accepted)

    rejected = df.loc[~ok, ["file_row"] + REQUIRED_COLUMNS].copy()
    rejected["reason"] = reason[~ok]
    return accepted, rejected, merged, None


# ---------------- saving, backup, undo ----------------
def save_upload(accepted):
    """Adds the accepted rows to the uploaded database (after making a backup). Returns (added, updated)."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    old = load_uploaded()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    old.to_csv(os.path.join(BACKUP_DIR, f"uploaded_monthly_{stamp}.csv"), index=False)   # the version BEFORE this upload

    new = accepted.copy()
    new["uploaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    combined = new if old.empty else pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "location", "commodity"], keep="last")   # newer value wins
    combined = combined.sort_values(["location", "commodity", "date"]).reset_index(drop=True)
    combined.to_csv(UPLOADED_PATH, index=False)

    updated = len(old) + len(new) - len(combined)
    return len(new) - updated, updated


def undo_last_upload():
    if not os.path.isdir(BACKUP_DIR):
        return False
    backups = sorted(f for f in os.listdir(BACKUP_DIR) if f.startswith("uploaded_monthly_"))
    if not backups:
        return False
    last = os.path.join(BACKUP_DIR, backups[-1])
    shutil.copy(last, UPLOADED_PATH)
    os.remove(last)
    return True


# ---------------- training a new model ----------------
def build_training_table(monthly):
    """Same features as notebook 06: this month's price, 3 lags, month number, time index -> next month's price."""
    df = monthly[REQUIRED_COLUMNS + ["source"]].sort_values(["location", "commodity", "date"]).reset_index(drop=True)
    grouped = df.groupby(["location", "commodity"])["price_per_kg"]
    df["month_num"] = df["date"].dt.month
    df["time_index"] = (df["date"].dt.year - 2025) * 12 + df["date"].dt.month - 1     # months since Jan 2025
    for lag in (1, 2, 3):
        df[f"lag{lag}"] = grouped.shift(lag)
    df["target_next_month"] = grouped.shift(-1)
    return df.dropna().reset_index(drop=True)


def _new_model():
    return xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)


def save_model(model, features, model_path=MODEL_PATH, features_path=FEATURES_PATH):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save_model(model_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(list(features), f, indent=1)


def load_model(model_path=MODEL_PATH, features_path=FEATURES_PATH):
    """Returns (model, feature_columns). Raises ValueError with a readable message if a file is unusable."""
    for path in (model_path, features_path):
        if not os.path.exists(path):
            raise ValueError(f"'{path}' is missing. Run notebooks/02_train_forecast_model.ipynb to create it.")
    try:
        model = _new_model()
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
    """Trains a NEW model, tests it on the latest months, and returns it together with the test results."""
    data = monthly[monthly["date"] <= current_month_start()]          # never train on future-dated months
    table = build_training_table(data)
    months = sorted(table["date"].drop_duplicates())
    if len(table) < 200 or len(months) < test_months + 3:
        raise ValueError("Not enough data to train and test a model.")

    is_test = table["date"].isin(months[-test_months:])
    enc = pd.get_dummies(table, columns=["location", "commodity"])
    feature_cols = [c for c in enc.columns if c not in ("date", "target_next_month", "source")]
    X, y = enc[feature_cols], enc["target_next_month"]

    test_model = _new_model().fit(X[~is_test], y[~is_test])            # 1) learn from the older months
    pred = test_model.predict(X[is_test])                              # 2) predict the latest months
    y_test = y[is_test]
    mae_model = float((y_test - pred).abs().mean())
    mae_baseline = float((y_test - X.loc[is_test, "price_per_kg"]).abs().mean())   # "next month = this month"
    mape_model = float(((y_test - pred).abs() / y_test).mean() * 100)

    final_model = _new_model().fit(X, y)                               # 3) the saved model learns from ALL months
    return {
        "model": final_model,
        "features": feature_cols,
        "metrics": {
            "rows": int(len(table)),
            "series": int(table.groupby(["location", "commodity"]).ngroups),
            "test_months": test_months,
            "test_rows": int(is_test.sum()),
            "real_test_rows": int((is_test & (table["source"] == "uploaded")).sum()),
            "mae_model": mae_model,
            "mae_baseline": mae_baseline,
            "mape_model": mape_model,
            "passed": mae_model <= mae_baseline,
        },
    }


def install_model(candidate):
    """Backs up the current model, then saves the new one."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(MODEL_BACKUP_DIR, stamp)
    os.makedirs(folder, exist_ok=True)
    for path in (MODEL_PATH, FEATURES_PATH):
        if os.path.exists(path):
            shutil.copy(path, folder)
    save_model(candidate["model"], candidate["features"])

    m = candidate["metrics"]
    entry = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows": m["rows"], "series": m["series"], "test_rows": m["test_rows"],
        "mae_model": round(m["mae_model"], 2), "mae_baseline": round(m["mae_baseline"], 2),
    }])
    if os.path.exists(TRAINING_LOG):
        entry = pd.concat([pd.read_csv(TRAINING_LOG), entry], ignore_index=True)
    entry.to_csv(TRAINING_LOG, index=False)


# ---------------- district prices (tab 2) ----------------
def load_district_prices():
    df = pd.read_csv(DISTRICT_PATH)
    df["Year"] = df["Year"].astype(int)
    return df


def common_commodities(district_df, district_a, district_b):
    """Commodities for which BOTH districts have a real price."""
    a = set(district_df.loc[district_df["location"] == district_a, "commodity"])
    b = set(district_df.loc[district_df["location"] == district_b, "commodity"])
    return sorted(a & b)


def cross_district_estimate(district_df, source, target, commodity):
    """Price-ratio method: how much more (or less) expensive is the target district than the source district?

    The ratio target/source is calculated for every year in which BOTH districts have a price, then averaged.
    """
    if source == target:
        raise ValueError("Please choose two different districts.")
    sub = district_df[(district_df["commodity"] == commodity) & (district_df["location"].isin([source, target]))]
    table = sub.pivot_table(index="location", columns="Year", values="price_per_kg")
    table = table.dropna(axis=1)                       # keep only years where BOTH districts have a price
    if table.shape[1] == 0 or source not in table.index or target not in table.index:
        raise ValueError("No year has a price in both districts for this commodity.")
    yearly_ratio = table.loc[target] / table.loc[source]
    return {
        "table": table,
        "yearly_ratio": yearly_ratio,
        "ratio": float(yearly_ratio.mean()),
        "typical_source": float(table.loc[source].mean()),
        "typical_target": float(table.loc[target].mean()),
    }


# ---------------- forecasting (tab 1) ----------------
def make_feature_row(features, prices, last_date, last_time_index, step, location, commodity):
    """One row of model inputs. It describes the month we are standing in; the model predicts the month after."""
    row = {col: 0 for col in features}
    row["price_per_kg"] = prices[-1]
    row["lag1"] = prices[-2]
    row["lag2"] = prices[-3]
    row["lag3"] = prices[-4]
    row["time_index"] = last_time_index + step
    row["month_num"] = (last_date + pd.DateOffset(months=step)).month
    for key in (f"location_{location}", f"commodity_{commodity}"):
        if key in row:
            row[key] = 1
    return row


def forecast_series(monthly, model, features, location, commodity, months_ahead, today=None, history_months=12):
    """Forecast a series month by month, always starting from NEXT month (relative to today)."""
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    current_month = pd.Timestamp(today.year, today.month, 1)

    hist = monthly[(monthly["location"] == location) & (monthly["commodity"] == commodity)
                   & (monthly["date"] <= current_month)].sort_values("date")
    if len(hist) < 4:
        raise ValueError("Not enough monthly history for this district and commodity.")

    prices = hist["price_per_kg"].tolist()
    last_date = hist["date"].max()
    last_time_index = (last_date.year - 2025) * 12 + (last_date.month - 1)      # months since Jan 2025
    # months between the last known month and the current month (0 while the data still covers today)
    skip = (current_month.year - last_date.year) * 12 + (current_month.month - last_date.month)

    rows = []
    for step in range(skip + months_ahead):
        row = make_feature_row(features, prices, last_date, last_time_index, step, location, commodity)
        price = float(model.predict(pd.DataFrame([row])[features])[0])
        if step >= skip:                                                         # hide months before next month
            rows.append({"date": last_date + pd.DateOffset(months=step + 1), "price": price})
        prices.append(price)

    return {
        "forecast": pd.DataFrame(rows),
        "history": hist.tail(history_months)[["date", "price_per_kg"]].reset_index(drop=True),
        "last_known": last_date,
        "source": str(hist["source"].iloc[-1]) if "source" in hist.columns else "generated",
    }


# ---------------- small helpers for the app ----------------
def data_signature():
    """Changes whenever a data file changes (used so the app never shows stale cached data)."""
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (GENERATED_PATH, UPLOADED_PATH, DISTRICT_PATH))


def model_signature():
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (MODEL_PATH, FEATURES_PATH))


def dataset_summary(monthly):
    used = monthly[monthly["date"] <= current_month_start()]
    real = used[used["source"] == "uploaded"] if "source" in used.columns else used.iloc[0:0]
    return {
        "districts": int(used["location"].nunique()),
        "series": int(used.groupby(["location", "commodity"]).ngroups),
        "latest_month": used["date"].max(),
        "real_series": int(real.groupby(["location", "commodity"]).ngroups),
    }
