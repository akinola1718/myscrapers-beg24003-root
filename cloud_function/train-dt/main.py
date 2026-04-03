# Decision Tree: train on all data < today (local TZ); hold out today
# HTTP entrypoint: train_dt_http

import os
import io
import json
import logging
import traceback
import numpy as np
import pandas as pd

from google.cloud import storage

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

# ---- ENV ----
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "structured/preds")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")


def _write_png_to_gcs(client: storage.Client, bucket: str, key: str, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(buf.read(), content_type="image/png")
    plt.close(fig)


def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def run_once(dry_run: bool = False, max_depth: int = 12, min_samples_leaf: int = 10):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numerics BEFORE counting/dropping ---
    orig_rows = len(df)
    df["price_num"] = _clean_numeric(df["price"])
    df["year_num"] = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    # --- Engineered features ---
    current_year = pd.Timestamp.utcnow().year
    df["vehicle_age"] = current_year - df["year_num"]
    df["vehicle_age"] = df["vehicle_age"].clip(lower=0)

    df["mileage_per_year"] = df["mileage_num"] / df["vehicle_age"].replace(0, np.nan)
    df["mileage_per_year"] = df["mileage_per_year"].replace([np.inf, -np.inf], np.nan)

    # --- Standardize text/categorical fields ---
    text_cols = [
        "make",
        "model",
        "transmission",
        "condition",
        "fuel",
        "color",
        "body_type",
        "title_status",
        "city",
        "state",
    ]

    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
            df.loc[df[c].isin(["nan", "none", "null", ""]), c] = np.nan

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | with valid numeric price=%d", orig_rows, valid_price_rows)

    counts = df["date_local"].value_counts().sort_index()
    logging.info(
        "Recent date counts (local): %s",
        json.dumps({str(k): int(v) for k, v in counts.tail(8).items()}),
    )

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {
            "status": "noop",
            "reason": "need at least two distinct dates",
            "dates": [str(d) for d in unique_dates],
        }

    today_local = unique_dates[-1]
    train_df = df[df["date_local"] < today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    train_df = train_df[train_df["price_num"].notna()]
    dropped_for_target = int((df["date_local"] < today_local).sum()) - int(len(train_df))
    logging.info("Train rows after target clean: %d (dropped_for_target=%d)", len(train_df), dropped_for_target)
    logging.info("Holdout rows today (%s): %d", today_local, len(holdout_df))

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    # --- Model features -> price_num ---
    target = "price_num"

    cat_cols = [
        "make",
        "model",
        "transmission",
        "condition",
        "fuel",
        "color",
        "body_type",
        "title_status",
    ]

    # Add these later if they are populated enough:
    # cat_cols += ["city", "state"]

    num_cols = [
        "year_num",
        "mileage_num",
        "vehicle_age",
        "mileage_per_year",
    ]

    feats = cat_cols + num_cols

    # Ensure all feature columns exist
    for col in feats:
        if col not in train_df.columns:
            train_df[col] = np.nan
        if col not in holdout_df.columns:
            holdout_df[col] = np.nan

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train = train_df[feats]
    y_train = train_df[target]
    pipe.fit(X_train, y_train)

    # ---- Predict/evaluate on today's holdout ----
    mae_today = None
    rmse_today = None
    mape_today = None
    bias_today = None
    preds_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    pi_df = pd.DataFrame()

    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    hour_key = now_utc.strftime("%Y%m%d%H")

    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat = pipe.predict(X_h)

        cols = ["post_id", "scraped_at", "make", "model", "year", "mileage", "price"]
        available_cols = [c for c in cols if c in holdout_df.columns]
        preds_df = holdout_df[available_cols].copy()
        preds_df["actual_price"] = holdout_df["price_num"]
        preds_df["pred_price"] = np.round(y_hat, 2)
        preds_df["abs_error"] = (preds_df["pred_price"] - preds_df["actual_price"]).abs()
        preds_df["pct_error"] = preds_df["abs_error"] / preds_df["actual_price"].replace(0, np.nan)

        if holdout_df["price_num"].notna().any():
            y_true = holdout_df["price_num"]
            mask = y_true.notna()

            if mask.any():
                mae_today = float(mean_absolute_error(y_true[mask], y_hat[mask]))
                rmse_today = float(np.sqrt(mean_squared_error(y_true[mask], y_hat[mask])))
                mape_today = float(mean_absolute_percentage_error(y_true[mask], y_hat[mask]))
                bias_today = float((y_hat[mask] - y_true[mask]).mean())

                metrics_df = pd.DataFrame(
                    [
                        {
                            "today_local": str(today_local),
                            "train_rows": int(len(train_df)),
                            "holdout_rows": int(len(holdout_df)),
                            "valid_price_rows": valid_price_rows,
                            "mae": mae_today,
                            "rmse": rmse_today,
                            "mape": mape_today,
                            "bias": bias_today,
                        }
                    ]
                )

                perm = permutation_importance(
                    pipe,
                    X_h[mask],
                    y_true[mask],
                    n_repeats=5,
                    random_state=42,
                    n_jobs=-1,
                )

                feature_names = pipe.named_steps["pre"].get_feature_names_out()
                pi_df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std,
                    }
                ).sort_values("importance_mean", ascending=False)

                top_pdp_features = ["mileage_num", "year_num", "vehicle_age"]
                for feat in top_pdp_features:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    PartialDependenceDisplay.from_estimator(pipe, X_train, [feat], ax=ax)
                    pdp_key = f"{OUTPUT_PREFIX}/{hour_key}/pdp_{feat}.png"
                    _write_png_to_gcs(client, GCS_BUCKET, pdp_key, fig)

    # --- Output paths ---
    preds_key = f"{OUTPUT_PREFIX}/{hour_key}/preds_llm.csv"
    metrics_key = f"{OUTPUT_PREFIX}/{hour_key}/metrics.csv"
    pi_key = f"{OUTPUT_PREFIX}/{hour_key}/permutation_importance.csv"

    if not dry_run and len(preds_df) > 0:
        _write_csv_to_gcs(client, GCS_BUCKET, preds_key, preds_df)
        logging.info("Wrote predictions to gs://%s/%s (%d rows)", GCS_BUCKET, preds_key, len(preds_df))

        if len(metrics_df) > 0:
            _write_csv_to_gcs(client, GCS_BUCKET, metrics_key, metrics_df)
            logging.info("Wrote metrics to gs://%s/%s", GCS_BUCKET, metrics_key)

        if len(pi_df) > 0:
            _write_csv_to_gcs(client, GCS_BUCKET, pi_key, pi_df)
            logging.info("Wrote permutation importance to gs://%s/%s", GCS_BUCKET, pi_key)
    else:
        logging.info("Dry run or no holdout rows; skip write. Would write to gs://%s/%s", GCS_BUCKET, preds_key)

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_price_rows": valid_price_rows,
        "mae_today": mae_today,
        "rmse_today": rmse_today,
        "mape_today": mape_today,
        "bias_today": bias_today,
        "predictions_key": preds_key,
        "metrics_key": metrics_key,
        "pi_key": pi_key,
        "dry_run": dry_run,
        "timezone": TIMEZONE,
    }


def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False)),
            max_depth=int(body.get("max_depth", 12)),
            min_samples_leaf=int(body.get("min_samples_leaf", 10)),
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
