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


def test_evaluate_current_model_reports_accuracy_without_changing_the_saved_model():
    monthly = dt.load_effective_monthly()
    model, features = dt.load_model()
    before_path_size = os.path.getsize(dt.MODEL_PATH)

    result = dt.evaluate_current_model(monthly, model, features, test_months=3)
    assert result["test_rows"] > 0
    assert result["mae_model"] >= 0 and result["mae_baseline"] >= 0
    assert isinstance(result["beats_baseline"], bool)
    assert os.path.getsize(dt.MODEL_PATH) == before_path_size          # evaluating never writes to the model file


def test_evaluate_current_model_needs_enough_data():
    model, features = dt.load_model()
    tiny = dt.load_effective_monthly().iloc[0:0]
    with pytest.raises(ValueError):
        dt.evaluate_current_model(tiny, model, features)


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


# ---------------------------------------------------------------- statistics and comparison
def test_every_commodity_has_exactly_one_category():
    districts = dt.load_district_prices()
    assert set(districts["category"]) == {"Chicken", "Rice", "Fish"}
    categories_per_commodity = districts.groupby("commodity")["category"].nunique()
    assert (categories_per_commodity == 1).all()


def test_commodity_stats_has_one_row_per_commodity_with_a_category():
    stats = dt.commodity_stats(dt.load_effective_monthly())
    districts = dt.load_district_prices()
    assert len(stats) == districts["commodity"].nunique()
    assert not stats["category"].isna().any()
    assert {"avg_2025", "avg_2026", "pct_change_2025_2026", "districts_covered"}.issubset(stats.columns)


def test_category_summary_covers_all_three_categories():
    summary = dt.category_summary()
    assert set(summary["category"]) == {"Chicken", "Rice", "Fish"}
    assert summary["commodities"].sum() == dt.load_district_prices()["commodity"].nunique()


def test_commodity_price_by_district_is_one_row_per_district_sorted_high_to_low():
    monthly = dt.load_effective_monthly()
    snapshot = dt.commodity_price_by_district(monthly, "Broiler chicken")
    assert not snapshot.empty
    assert not snapshot["location"].duplicated().any()
    assert snapshot["price_per_kg"].is_monotonic_decreasing


# ---------------------------------------------------------------- budget planner
def test_budget_opportunities_excludes_fish_by_default():
    monthly = dt.load_effective_monthly()
    model, features = dt.load_model()
    out = dt.budget_opportunities(monthly, model, features, budget=1000, months_ahead=6, location="Dhaka")
    assert not out.empty
    assert "Fish" not in set(out["category"])


def test_budget_opportunities_only_lists_whats_affordable_and_ranks_by_predicted_change():
    monthly = dt.load_effective_monthly()
    model, features = dt.load_model()
    out = dt.budget_opportunities(monthly, model, features, budget=60, months_ahead=6, location="Dhaka")
    assert (out["current_price"] <= 60).all()
    assert out["predicted_change_pct"].is_monotonic_decreasing


def test_budget_opportunities_rejects_a_non_positive_budget():
    monthly = dt.load_effective_monthly()
    model, features = dt.load_model()
    with pytest.raises(ValueError):
        dt.budget_opportunities(monthly, model, features, budget=0, months_ahead=6, location="Dhaka")


# ---------------------------------------------------------------- login (no password)
def test_a_new_identifier_creates_an_account(monkeypatch, tmp_path):
    monkeypatch.setattr(dt, "USERS_PATH", str(tmp_path / "users.csv"))
    identifier, is_new = dt.login_or_register("Rafi Uddin")
    assert identifier == "Rafi Uddin" and is_new is True


def test_the_same_identifier_logs_back_into_the_same_account(monkeypatch, tmp_path):
    monkeypatch.setattr(dt, "USERS_PATH", str(tmp_path / "users.csv"))
    dt.login_or_register("Rafi Uddin")
    identifier, is_new = dt.login_or_register("Rafi Uddin")
    assert identifier == "Rafi Uddin" and is_new is False
    assert len(dt.load_users()) == 1                            # no duplicate account was created


def test_an_empty_identifier_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(dt, "USERS_PATH", str(tmp_path / "users.csv"))
    with pytest.raises(ValueError):
        dt.login_or_register("   ")


# ---------------------------------------------------------------- anomaly detection
def _dhaka_broiler():
    monthly = dt.load_effective_monthly()
    sub = monthly[(monthly["location"] == "Dhaka") & (monthly["commodity"] == "Broiler chicken")]
    last_price = float(sub.sort_values("date")["price_per_kg"].iloc[-1])
    return last_price, monthly


def test_a_price_within_the_accept_limits_is_not_unusual():
    last_price, monthly = _dhaka_broiler()
    # 30% rise and 15% fall are exactly at the accept limit - a hair under each should still be accepted.
    just_under_rise = dt.assess_submission(monthly, "Dhaka", "Broiler chicken", round(last_price * 1.29, 2))
    just_under_fall = dt.assess_submission(monthly, "Dhaka", "Broiler chicken", round(last_price * 0.86, 2))
    assert just_under_rise["unusual"] is False and just_under_rise["reason"] is None
    assert just_under_fall["unusual"] is False and just_under_fall["reason"] is None
    assert just_under_rise["reference_price"] == pytest.approx(last_price)


def test_a_rise_or_fall_just_past_its_accept_limit_is_moderate():
    last_price, monthly = _dhaka_broiler()
    rise = dt.assess_submission(monthly, "Dhaka", "Broiler chicken", round(last_price * 1.35, 2))    # 35% rise: past the 30% rise limit, short of the 50% severe mark
    fall = dt.assess_submission(monthly, "Dhaka", "Broiler chicken", round(last_price * 0.80, 2))     # 20% fall: past the 15% fall limit, short of the 30% severe mark
    assert rise["unusual"] is True and rise["severity"] == "moderate" and "rise" in rise["reason"]
    assert fall["unusual"] is True and fall["severity"] == "moderate" and "drop" in fall["reason"]


def test_a_rise_or_fall_past_the_severe_limit_is_labelled_severe():
    last_price, monthly = _dhaka_broiler()
    rise = dt.assess_submission(monthly, "Dhaka", "Broiler chicken", round(last_price * 1.6, 2))     # 60% rise: past the 50% severe mark
    fall = dt.assess_submission(monthly, "Dhaka", "Broiler chicken", round(last_price * 0.6, 2))      # 40% fall: past the 30% severe mark
    assert rise["unusual"] is True and rise["severity"] == "severe"
    assert fall["unusual"] is True and fall["severity"] == "severe"


def test_the_fall_limit_is_stricter_than_the_rise_limit():
    """A 20% move is accepted as a rise but flagged as a fall - rise and fall use different limits on purpose."""
    last_price, monthly = _dhaka_broiler()
    rise_20 = dt.assess_submission(monthly, "Dhaka", "Broiler chicken", round(last_price * 1.20, 2))
    fall_20 = dt.assess_submission(monthly, "Dhaka", "Broiler chicken", round(last_price * 0.80, 2))
    assert rise_20["unusual"] is False
    assert fall_20["unusual"] is True


def test_a_series_with_no_prior_data_is_always_unusual():
    monthly = dt.load_effective_monthly()
    empty_monthly = monthly.iloc[0:0]
    result = dt.assess_submission(empty_monthly, "Dhaka", "Broiler chicken", 180)
    assert result["unusual"] is True and result["reference_price"] is None and result["severity"] == "moderate"


# ---------------------------------------------------------------- public price submissions
def test_a_normal_price_is_added_immediately_and_earns_a_point(sandbox, points_sandbox, monkeypatch):
    monkeypatch.setattr(dt, "COMMUNITY_PATH", str(sandbox / "community.csv"))
    last_price, monthly = _dhaka_broiler()
    locs, coms, medians = dt.known_names()

    ok, status, info = dt.submit_community_price("2026-08-01", "Dhaka", "Broiler chicken",
                                                 round(last_price * 1.05, 2), "Rafi", monthly, locs, coms, medians)
    assert ok and status == "auto_approved" and info["unusual"] is False
    assert len(dt.load_uploaded()) == 1
    assert dt.pending_community_submissions().empty                # never needed review
    assert dt.contributor_status("Rafi")["points_balance"] == 1    # point credited right away


def test_an_unusual_price_is_held_and_earns_no_point_until_approved(sandbox, points_sandbox, monkeypatch):
    monkeypatch.setattr(dt, "COMMUNITY_PATH", str(sandbox / "community.csv"))
    last_price, monthly = _dhaka_broiler()
    locs, coms, medians = dt.known_names()

    ok, status, info = dt.submit_community_price("2026-08-01", "Dhaka", "Broiler chicken",
                                                 round(last_price * 2, 2), "Rafi", monthly, locs, coms, medians)
    assert ok and status == "pending" and info["unusual"] is True
    assert len(dt.load_uploaded()) == 0                             # not added yet
    assert dt.contributor_status("Rafi")["points_balance"] == 0     # no point yet

    pending = dt.pending_community_submissions()
    assert len(pending) == 1 and pending.iloc[0]["contributor"] == "Rafi"
    dt.review_community_submission(int(pending.iloc[0]["row_id"]), approve=True)
    assert dt.pending_community_submissions().empty
    assert len(dt.load_uploaded()) == 1
    assert dt.contributor_status("Rafi")["points_balance"] == 1


def test_submission_without_a_contributor_name_is_rejected():
    last_price, monthly = _dhaka_broiler()
    locs, coms, medians = dt.known_names()
    ok, status, message = dt.submit_community_price("2026-08-01", "Dhaka", "Broiler chicken", last_price,
                                                     "  ", monthly, locs, coms, medians)
    assert not ok and status is None and "log in" in message


def test_invalid_submission_is_rejected_with_a_reason_and_never_queued(sandbox, monkeypatch):
    monkeypatch.setattr(dt, "COMMUNITY_PATH", str(sandbox / "community.csv"))
    last_price, monthly = _dhaka_broiler()
    locs, coms, medians = dt.known_names()

    ok, status, message = dt.submit_community_price("2099-01-01", "Dhaka", "Broiler chicken", last_price,
                                                     "Rafi", monthly, locs, coms, medians)
    assert not ok and status is None and "future" in message
    assert dt.pending_community_submissions().empty


def test_rejecting_an_unusual_submission_leaves_the_uploaded_database_untouched(sandbox, points_sandbox, monkeypatch):
    monkeypatch.setattr(dt, "COMMUNITY_PATH", str(sandbox / "community.csv"))
    last_price, monthly = _dhaka_broiler()
    locs, coms, medians = dt.known_names()
    dt.submit_community_price("2026-08-01", "Dhaka", "Broiler chicken", round(last_price * 2, 2),
                              "Rafi", monthly, locs, coms, medians)

    pending = dt.pending_community_submissions()
    dt.review_community_submission(int(pending.iloc[0]["row_id"]), approve=False)
    assert dt.pending_community_submissions().empty
    assert len(dt.load_uploaded()) == 0
    assert dt.load_community_submissions().iloc[0]["status"] == "rejected"
    assert dt.contributor_status("Rafi")["points_balance"] == 0


# ---------------------------------------------------------------- points ledger
@pytest.fixture
def points_sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(dt, "POINTS_PATH", str(tmp_path / "points.csv"))
    return tmp_path


def test_new_contributor_has_the_free_trials_and_no_points(points_sandbox):
    status = dt.contributor_status("New Person")
    assert status == {"free_remaining": dt.FREE_BUDGET_PLANS, "points_balance": 0}


def test_free_trials_run_out_before_points_are_needed(points_sandbox):
    for _ in range(dt.FREE_BUDGET_PLANS):
        assert dt.consume_budget_plan_credit("Rafi") == "free_trial"
    with pytest.raises(ValueError, match="No free trials or points"):
        dt.consume_budget_plan_credit("Rafi")


def test_a_point_buys_exactly_one_run_after_free_trials_are_used(points_sandbox):
    for _ in range(dt.FREE_BUDGET_PLANS):
        dt.consume_budget_plan_credit("Rafi")
    dt.award_point("Rafi")
    assert dt.contributor_status("Rafi")["points_balance"] == 1
    assert dt.consume_budget_plan_credit("Rafi") == "point"
    assert dt.contributor_status("Rafi")["points_balance"] == 0
    with pytest.raises(ValueError):
        dt.consume_budget_plan_credit("Rafi")


def test_points_are_tracked_separately_per_contributor(points_sandbox):
    for _ in range(dt.FREE_BUDGET_PLANS):
        dt.consume_budget_plan_credit("Rafi")
    assert dt.contributor_status("Rafi")["free_remaining"] == 0
    assert dt.contributor_status("Someone Else")["free_remaining"] == dt.FREE_BUDGET_PLANS


def test_leaderboard_ranks_by_points_and_skips_zero_balances(points_sandbox):
    dt.award_point("Rafi")
    dt.award_point("Rafi")
    dt.award_point("Priya")
    dt.consume_budget_plan_credit("Zero Points Zahid")           # uses a free trial only - never earns a point

    board = dt.points_leaderboard()
    assert list(board["contributor"]) == ["Rafi", "Priya"]
    assert list(board["points_balance"]) == [2, 1]
    assert "Zero Points Zahid" not in set(board["contributor"])


# ---------------------------------------------------------------- submission statistics (Admin > Results)
def test_submission_stats_are_all_zero_with_no_submissions(sandbox, monkeypatch):
    monkeypatch.setattr(dt, "COMMUNITY_PATH", str(sandbox / "community.csv"))
    stats = dt.submission_stats()
    assert stats == {"total": 0, "auto_approved": 0, "pending": 0, "approved": 0, "rejected": 0,
                     "moderate": 0, "severe": 0}


def test_submission_stats_count_each_outcome(sandbox, points_sandbox, monkeypatch):
    monkeypatch.setattr(dt, "COMMUNITY_PATH", str(sandbox / "community.csv"))
    monthly = dt.load_effective_monthly()
    locs, coms, medians = dt.known_names()

    def last_price_in(location):
        sub = monthly[(monthly["location"] == location) & (monthly["commodity"] == "Broiler chicken")]
        return float(sub.sort_values("date")["price_per_kg"].iloc[-1])

    # one normal submission -> auto_approved
    dt.submit_community_price("2026-08-01", "Dhaka", "Broiler chicken",
                              round(last_price_in("Dhaka") * 1.05, 2), "Rafi", monthly, locs, coms, medians)
    # one moderately unusual submission, approved by an admin
    dt.submit_community_price("2026-08-01", "Chattogram", "Broiler chicken",
                              round(last_price_in("Chattogram") * 1.35, 2), "Rafi", monthly, locs, coms, medians)
    pending = dt.pending_community_submissions()
    dt.review_community_submission(int(pending.iloc[0]["row_id"]), approve=True)
    # one severely unusual submission, rejected by an admin
    dt.submit_community_price("2026-08-01", "Bogura", "Broiler chicken",
                              round(last_price_in("Bogura") * 1.6, 2), "Rafi", monthly, locs, coms, medians)
    pending2 = dt.pending_community_submissions()
    dt.review_community_submission(int(pending2.iloc[0]["row_id"]), approve=False)

    stats = dt.submission_stats()
    assert stats["total"] == 3
    assert stats["auto_approved"] == 1
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["pending"] == 0
    assert stats["moderate"] == 1        # the one that was approved
    assert stats["severe"] == 1          # the one that was rejected
