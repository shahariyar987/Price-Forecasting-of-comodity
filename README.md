# Bangladesh Commodity Price Predictor

A Streamlit web app that **forecasts retail prices** and **estimates prices across districts** for chicken, rice and fish
in Bangladesh. Administrators can upload real monthly prices and retrain the forecast model from inside the app.

## Features

| Tab | What it does |
|---|---|
| **Forecast** | Predicts the price of a commodity in a district for 1 to 18 months, starting next month. Shows a chart, a downloadable table and the data basis. |
| **Cross-district** | Enter the known price in one district and get an estimate for another district (price-ratio method). |
| **Admin** (password) | Upload monthly prices as CSV with automatic checks, undo an upload, retrain the forecast model with a safe test before replacing it, and see the numbers behind each result. |

## Quick start

Python 3.12 or newer is recommended.

```bash
python -m venv venv
venv\Scripts\activate            # Windows      (macOS / Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Run the last command from the project folder (the folder that contains `app.py`).

### Admin password (one-time setup)

The Admin tab stays locked until you set a password. Copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml`, choose your own password, and restart the app:

```toml
ADMIN_PASSWORD = "your-password"
```

`secrets.toml` is listed in `.gitignore`, so it is never committed. You can also set an `ADMIN_PASSWORD`
environment variable instead.

## How it works

**Data.** The raw files (`data/raw`) come from the Department of Agricultural Marketing (DAM) and contain
*one retail price per market per year* (2025 and 2026). They are cleaned and averaged per district
(`data/processed/district_prices_clean.csv`).

**Forecast.** The DAM files have no monthly prices, so a monthly series is *generated* from the two yearly averages
(a straight line plus small random noise, `data/processed/monthly_hybrid.csv`). An XGBoost model learns to predict
next month's price from this month's price, the three months before, the month number and the district/commodity.
Longer forecasts feed the model's own predictions back in, so they become less reliable the further ahead they go.

**Real data replaces generated data.** When an administrator uploads at least **6 consecutive months** of real prices
for a district and commodity, that series uses only the real prices from then on. Everything else keeps using the
generated series. The app always shows which one a forecast is based on.

**Cross-district.** For every year in which both districts have a price, the target price is divided by the source price.
The average of these ratios is multiplied by the price you enter. It uses only the real yearly prices.

## Admin tab: turning it on

The Admin tab is locked until a password exists. Set one **once**:

1. In the project folder, open `.streamlit/secrets.toml.example`.
2. Save a copy in the same folder named exactly `secrets.toml` (not `secrets.toml.txt`).
3. Put one line inside it, with your own password in the quotes:
   ```toml
   ADMIN_PASSWORD = "your-password"
   ```
4. Restart the app (`Ctrl+C`, then `streamlit run app.py`) and open the Admin tab.

Instead of the file, you can set an environment variable named `ADMIN_PASSWORD` before starting the app.
`secrets.toml` is listed in `.gitignore`, so it is never shared.

While logged in, "Details (administrators only)" sections appear in the Forecast and Cross-district tabs showing the
numbers behind each result. They disappear again when you press **Lock**.

## Admin tab: uploading data

Upload a CSV with the columns `date, location, commodity, price_per_kg` (see the *File format* sub-tab in the app).
Every row is checked before anything is saved: date format, future months, prices, and district and commodity names
(wrong spellings are rejected with a suggestion). Saving makes a backup first, and **Undo last upload** restores it.
**Retrain model** trains a new model, tests it on the latest 3 months against the baseline "next month = this month",
and only replaces the current model when you confirm. The old model is kept in `models/backups/`.

## Project structure

```
app.py                      the user interface (Streamlit)
data_tools.py               all calculations: data loading, upload checks, forecasting, training
requirements.txt            packages needed to run the app
requirements-dev.txt        extra packages for the notebooks and the tests
.streamlit/config.toml      theme and server settings
data/raw/                   DAM price files (chicken, rice, fish)
data/processed/             cleaned district prices and the generated monthly series
models/                     the trained forecast model (JSON) and its list of input columns
notebooks/                  01_data_preparation.ipynb, 02_train_forecast_model.ipynb
tests/                      automated tests
```

## Rebuilding the data and the model

```bash
pip install -r requirements-dev.txt
jupyter notebook
```

Run `notebooks/01_data_preparation.ipynb` first, then `notebooks/02_train_forecast_model.ipynb`.

## Tests

```bash
python -m pytest -q
```

The tests cover the upload checks, the real-versus-generated rule, undo, the forecast (including that it always starts
next month and uses exactly the inputs the model was trained on), the cross-district estimate, and the data files.

## Troubleshooting

**"The model files could not be read" or "is missing"** - rebuild them:
`pip install -r requirements-dev.txt`, open `notebooks/02_train_forecast_model.ipynb` and run *Kernel > Restart & Run All*.
This takes a few seconds and needs no download.

**The Admin tab says access is not set up** - create `.streamlit/secrets.toml` as described above and restart the app.

## Known limitations

- The monthly history is generated from yearly averages, so it contains no real month-to-month movement or seasonality.
  Forecasts are therefore only indicative until real monthly prices are uploaded.
- The Cross-district tab assumes the price gap between two districts stays the same; the ratio changed between 2025 and 2026.
- Tree-based models such as XGBoost cannot extrapolate beyond the price range they have seen.
- Predictions can be wrong and must not be the only basis for a decision.
