"""Automated checks for the Bangladesh Commodity Price Predictor.

Run from the project folder with:   python -m pytest -q
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import data_tools as dt  # noqa: E402


@pytest.fixture(autouse=True)
def project_folder(monkeypatch):
    """data_tools uses paths relative to the project folder."""
    monkeypatch.chdir(ROOT)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Uploads and backups go to a temporary folder, so the real data is never touched."""
    monkeypatch.setattr(dt, "UPLOADED_PATH", str(tmp_path / "uploaded.csv"))
    monkeypatch.setattr(dt, "BACKUP_DIR", str(tmp_path / "backups"))
    return tmp_path


def read(csv_text):
    return pd.read_csv(io.StringIO(csv_text), dtype=str)


def months_frame(location, commodity, months, start="2026-01-01", base=180.0):
    dates = pd.date_range(start, periods=months, freq="MS")
    return pd.DataFrame({"date": dates, "location": location, "commodity": commodity,
                         "price_per_kg": [round(base + i, 2) for i in range(months)]})


# ---------------------------------------------------------------- checking uploads
def test_validation_accepts_good_rows_and_rejects_bad_ones():
    csv_text = """date,location,commodity,price_per_kg
2026-05-01,Dhaka,Broiler chicken,180.1
2026-06-01,Dhaka,Broiler chicken,181.0
2026-08-01,Chittagong,Broiler chicken,170
2026-08-01,Chattogram,Broiler chicken,5000
2027-12-01,Dhaka,Broiler chicken,180
8/1/2026,Dhaka,Broiler chicken,180
2026-08-01,Dhaka,Broiler chicken,-5
2026-08-01,Dhaka,Hilsha,900
"""
    locs, coms, medians = dt.known_names()
    accepted, rejected, merged, error = dt.validate_upload(read(csv_text), locs, coms, medians)
    assert error is None
    assert len(accepted) == 2 and len(rejected) == 6
    reasons = " | ".join(rejected["reason"])
    for expected in ("did you mean 'Chattogram'", "3x away", "in the future", "date must look like",
                     "above 0", "did you mean 'Hilsha (500-900) gm'"):
        assert expected in reasons


def test_names_ignore_case_and_duplicates_are_averaged():
    csv_text = """Date,LOCATION,Commodity,price_per_kg
2026-05-10,dhaka,broiler chicken,180
2026-05-20,DHAKA ,Broiler chicken,190
"""
    locs, coms, medians = dt.known_names()
    accepted, rejected, merged, error = dt.validate_upload(read(csv_text), locs, coms, medians)
    assert len(rejected) == 0 and merged == 1
    assert accepted.iloc[0]["location"] == "Dhaka" and accepted.iloc[0]["price_per_kg"] == 185.0
    assert accepted.iloc[0]["date"] == pd.Timestamp("2026-05-01")


def test_new_names_need_permission():
    csv_text = "date,location,commodity,price_per_kg\n2026-05-01,Dhaka,Brand New Fish,250\n"
    locs, coms, medians = dt.known_names()
    _, rejected, _, _ = dt.validate_upload(read(csv_text), locs, coms, medians)
    assert len(rejected) == 1
    accepted, _, _, _ = dt.validate_upload(read(csv_text), locs, coms, medians, allow_new=True)
    assert len(accepted) == 1


def test_missing_columns_give_a_clear_error():
    locs, coms, medians = dt.known_names()
    accepted, rejected, merged, error = dt.validate_upload(read("date,price\n2026-01-01,5\n"), locs, coms, medians)
    assert accepted is None and "Missing column" in error


# ---------------------------------------------------------------- real data replaces generated data
def key_source(effective, location, commodity):
    sub = effective[(effective["location"] == location) & (effective["commodity"] == commodity)]
    return set(sub["source"])


def test_series_switches_to_real_data_after_enough_consecutive_months(sandbox):
    dt.save_upload(months_frame("Dhaka", "Broiler chicken", dt.MIN_REAL_MONTHS - 1))
    assert key_source(dt.load_effective_monthly(), "Dhaka", "Broiler chicken") == {"generated"}

    dt.save_upload(months_frame("Dhaka", "Broiler chicken", dt.MIN_REAL_MONTHS))
    effective = dt.load_effective_monthly()
    assert key_source(effective, "Dhaka", "Broiler chicken") == {"uploaded"}
    assert key_source(effective, "Chattogram", "Broiler chicken") == {"generated"}     # other series untouched


def test_a_missing_month_breaks_the_run(sandbox):
    frame = months_frame("Dhaka", "Broiler chicken", 8)
    dt.save_upload(frame.drop(index=3))                        # months 1-3, then 5-8 -> longest recent run is 4
    status = dt.uploaded_series_status().iloc[0]
    assert status["consecutive_months"] == 4 and status["status"].startswith("waiting")
    assert key_source(dt.load_effective_monthly(), "Dhaka", "Broiler chicken") == {"generated"}


def test_saving_the_same_rows_twice_does_not_duplicate_and_undo_works(sandbox):
    frame = months_frame("Dhaka", "Broiler chicken", 6)
    assert dt.save_upload(frame) == (6, 0)
    assert dt.save_upload(frame) == (0, 6)                     # same rows: updated, not added
    assert len(dt.load_uploaded()) == 6
    assert dt.undo_last_upload() is True and len(dt.load_uploaded()) == 6
    assert dt.undo_last_upload() is True and len(dt.load_uploaded()) == 0
    assert dt.undo_last_upload() is False


# ---------------------------------------------------------------- forecasting
def test_forecast_inputs_are_exactly_what_the_model_was_trained_on():
    monthly = dt.load_effective_monthly()
    table = dt.build_training_table(monthly)
    _, features = dt.load_model()
    checked = 0
    for (loc, com), series in list(monthly.groupby(["location", "commodity"]))[:40]:
        series = series.sort_values("date").reset_index(drop=True)
        train_rows = table[(table["location"] == loc) & (table["commodity"] == com)].sort_values("date")
        for _, train_row in train_rows.iloc[[0, len(train_rows) // 2, -1]].iterrows():
            t = series.index[series["date"] == train_row["date"]][0]
            prices = series["price_per_kg"].iloc[: t + 1].tolist()
            last_date = series["date"].iloc[t]
            time_index = (last_date.year - 2025) * 12 + last_date.month - 1
            row = dt.make_feature_row(features, prices, last_date, time_index, 0, loc, com)
            for name in ("price_per_kg", "lag1", "lag2", "lag3", "time_index", "month_num"):
                assert row[name] == pytest.approx(train_row[name])
            checked += 1
    assert checked > 100


@pytest.mark.parametrize("today, first_month", [("2026-09-19", "2026-10-01"), ("2026-12-31", "2027-01-01"),
                                                ("2027-03-10", "2027-04-01")])
def test_forecast_always_starts_next_month(today, first_month):
    monthly = dt.load_effective_monthly()
    model, features = dt.load_model()
    result = dt.forecast_series(monthly, model, features, "Dhaka", "Broiler chicken", 7, today=today)
    forecast = result["forecast"]
    assert forecast["date"].iloc[0] == pd.Timestamp(first_month)
    assert len(forecast) == 7 and np.isfinite(forecast["price"]).all()
    assert forecast["date"].is_monotonic_increasing


def test_model_files_are_text_and_reload_identically():
    """The model is stored as JSON so that copying or unzipping cannot silently damage it."""
    model, features = dt.load_model()
    with open(dt.MODEL_PATH, encoding="utf-8") as f:
        assert f.read(1) == "{"                                  # readable JSON, not a binary blob
    rng = np.random.RandomState(0).rand(5, len(features))
    again, _ = dt.load_model()
    assert np.allclose(model.predict(rng), again.predict(rng))


def test_missing_model_gives_a_readable_message(tmp_path):
    with pytest.raises(ValueError, match="rebuild|create"):
        dt.load_model(str(tmp_path / "nope.json"), str(tmp_path / "nope_features.json"))


def test_damaged_model_gives_a_readable_message(tmp_path):
    damaged = tmp_path / "damaged.json"
    damaged.write_text("{ this is not a model")
    features_copy = tmp_path / "features.json"
    features_copy.write_text(open(dt.FEATURES_PATH, encoding="utf-8").read())
    with pytest.raises(ValueError, match="damaged"):
        dt.load_model(str(damaged), str(features_copy))


def test_forecast_needs_enough_history():
    monthly = dt.load_effective_monthly()
    model, features = dt.load_model()
    with pytest.raises(ValueError):
        dt.forecast_series(monthly, model, features, "Dhaka", "Broiler chicken", 3, today="2025-02-10")


def test_saved_model_matches_its_feature_list():
    model, features = dt.load_model()
    assert model.n_features_in_ == len(features)
    assert features[:6] == ["price_per_kg", "month_num", "time_index", "lag1", "lag2", "lag3"]


# ---------------------------------------------------------------- cross-district estimate
def test_cross_district_ratio_uses_only_years_with_both_prices():
    districts = pd.DataFrame({
        "Year": [2025, 2026, 2026],
        "location": ["A", "A", "B"],
        "commodity": ["Rice"] * 3,
        "price_per_kg": [50.0, 60.0, 66.0],
    })
    result = dt.cross_district_estimate(districts, "A", "B", "Rice")
    assert result["ratio"] == pytest.approx(1.1)                # only 2026: 66 / 60
    assert result["typical_source"] == pytest.approx(60.0)
    with pytest.raises(ValueError):
        dt.cross_district_estimate(districts, "A", "A", "Rice")
    no_overlap = pd.DataFrame({"Year": [2025, 2026], "location": ["A", "B"],
                               "commodity": ["Rice", "Rice"], "price_per_kg": [50.0, 66.0]})
    with pytest.raises(ValueError):                             # A has 2025 only, B has 2026 only
        dt.cross_district_estimate(no_overlap, "A", "B", "Rice")


def test_common_commodities():
    districts = pd.DataFrame({"Year": [2026] * 4, "location": ["A", "A", "B", "B"],
                              "commodity": ["Rice", "Fish", "Rice", "Egg"], "price_per_kg": [1.0] * 4})
    assert dt.common_commodities(districts, "A", "B") == ["Rice"]


# ---------------------------------------------------------------- the data files
def test_monthly_data_is_clean():
    monthly = dt.load_generated()
    assert monthly[["date", "location", "commodity", "price_per_kg"]].notna().all().all()
    assert not monthly.duplicated(["location", "commodity", "date"]).any()
    assert (monthly["price_per_kg"] > 0).all()
    assert monthly["date"].max() <= dt.current_month_start(), "the file must not contain future months"
    for _, series in monthly.groupby(["location", "commodity"]):
        months = series["date"].dt.year * 12 + series["date"].dt.month
        assert (months.sort_values().diff().dropna() == 1).all(), "months must be consecutive"


def test_district_data_is_clean():
    districts = dt.load_district_prices()
    assert set(districts["Year"]) == {2025, 2026}
    assert (districts["price_per_kg"] > 0).all()
    assert not districts.duplicated(["Year", "location", "commodity"]).any()
    assert districts["price_per_kg"].max() < 3000                # typos such as 12,157 Tk/kg were removed
