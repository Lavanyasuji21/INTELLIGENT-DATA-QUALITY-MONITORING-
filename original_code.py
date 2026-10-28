# ================================================================
# INTELLIGENT DATA QUALITY MONITORING PLATFORM
# Complete Google Colab Application - Single Cell
#
# FEATURES:
# 1. Data Quality Analysis
# 2. Random Forest Regression / Classification for Missing Values
# 3. Human-in-the-Loop Approval and Manual Override
# 4. Duplicate Handling
# 5. Explainable Autoencoder Anomaly Detection
# 6. Isolation Forest Anomaly Detection
# 7. PSI + KS Test + Chi-Square Data Drift Detection
# 8. Before-and-After Quality Comparison
# 9. Excel Reports and Dataset Downloads
# 10. Adjustable Autoencoder Threshold (Default: 95th Percentile)
# ================================================================

# %pip -q install gradio pandas numpy matplotlib scikit-learn tensorflow openpyxl xlrd scipy

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import ks_2samp, chi2_contingency
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================================================================
# CONFIGURATION
# ================================================================

REPORT_DIR = "/content/data_quality_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

AE_THRESHOLD_PERCENTILE = 95

APP = {
    "original": None,
    "proposed": None,
    "approved": None,
    "issues": None,
    "quality_summary": None,
    "imputation_report": None,
    "anomaly_report": None,
    "drift_report": None,
    "reference_data": None,
    "current_data": None,
    "quality_comparison": None,
    "anomaly_plot": None,
    "drift_plot": None,
    "quality_comparison_plot": None,
    "quality_report_path": None,
    "anomaly_report_path": None,
    "drift_report_path": None,
    "combined_report_path": None
}


# ================================================================
# GENERAL UTILITIES
# ================================================================

def read_dataset(file):
    if file is None:
        raise ValueError("Please upload a dataset first.")

    path = file.name if hasattr(file, "name") else str(file)
    extension = os.path.splitext(path)[1].lower()

    if extension == ".csv":
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Supported formats are CSV, XLS and XLSX.")

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    df.columns = [
        str(c).strip() if str(c).strip() else f"Unnamed_{i}"
        for i, c in enumerate(df.columns)
    ]

    return df


def safe_string(value):
    if pd.isna(value):
        return ""
    return str(value)


def numeric_columns(df):
    return list(df.select_dtypes(include=np.number).columns)


def categorical_columns(df):
    return list(df.select_dtypes(exclude=np.number).columns)


def is_identifier_column(column_name):
    name = str(column_name).strip().lower()
    normalized = name.replace("-", "_").replace(" ", "_")
    tokens = normalized.split("_")

    identifiers = {
        "id", "rowid", "index", "recordid", "customerid",
        "studentid", "userid", "transactionid", "uuid"
    }

    if normalized in identifiers:
        return True

    if any(t in {"target", "label", "class", "outcome"} for t in tokens):
        return True

    if normalized in {"timestamp", "datetime", "date_time", "date"}:
        return True

    return False


def make_excel_report(sheets, output_path):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        written = False

        for sheet_name, data in sheets.items():
            if data is None:
                continue

            export_df = (
                data.copy() if isinstance(data, pd.DataFrame)
                else pd.DataFrame(data)
            )

            export_df.to_excel(
                writer,
                sheet_name=str(sheet_name)[:31],
                index=False
            )
            written = True

        if not written:
            pd.DataFrame({"Message": ["No report data available."]}).to_excel(
                writer, sheet_name="Report", index=False
            )

    return output_path


def dataframe_preview(df, rows=15):
    if df is None:
        return pd.DataFrame()
    return df.head(rows).copy()


# ================================================================
# RANDOM FOREST IMPUTATION
# ================================================================

def prepare_predictors(df, target_column):
    X = df.drop(columns=[target_column]).copy()

    if X.shape[1] == 0:
        return None

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
            median = X[col].median()
            X[col] = X[col].fillna(0 if pd.isna(median) else median)
        else:
            X[col] = (
                X[col].astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

    X = pd.get_dummies(X, dummy_na=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X.astype(float) if not X.empty else None


def predict_missing_column(df, target_column):
    result = df[target_column].copy()
    missing_mask = result.isna()

    if not missing_mask.any():
        return result, "No missing values"

    observed = result.dropna()

    if observed.empty:
        return result, "Not imputed: column contains only missing values"

    is_numeric = pd.api.types.is_numeric_dtype(result)

    if is_numeric:
        fallback = result.mean()
        method = "Mean fallback"
    else:
        modes = observed.mode()
        fallback = modes.iloc[0] if not modes.empty else observed.iloc[0]
        method = "Mode fallback"

    result.loc[missing_mask] = fallback

    if len(observed) < 10 or df.shape[1] < 2:
        return result, method

    try:
        X_all = prepare_predictors(df, target_column)

        if X_all is None:
            return result, method

        observed_mask = df[target_column].notna()
        X_train = X_all.loc[observed_mask]
        X_missing = X_all.loc[missing_mask]
        y_train = df.loc[observed_mask, target_column]

        if len(X_train) < 10 or X_missing.empty:
            return result, method

        if is_numeric:
            y_train = pd.to_numeric(y_train, errors="coerce")
            valid_y = y_train.notna()
            X_train = X_train.loc[valid_y]
            y_train = y_train.loc[valid_y]

            if len(y_train) < 10:
                return result, method

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            result.loc[missing_mask] = model.predict(X_missing)
            method = "Random Forest Regression"

        else:
            y_train = y_train.astype(str)

            if y_train.nunique() < 2:
                return result, method

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
            model.fit(X_train, y_train)
            result.loc[missing_mask] = model.predict(X_missing)
            method = "Random Forest Classification"

    except Exception as exc:
        method = f"{method} (model fallback: {type(exc).__name__})"

    return result, method


# ================================================================
# DATA QUALITY ANALYSIS
# ================================================================

def calculate_quality_metrics(df):
    rows, cols = df.shape
    total_cells = max(rows * cols, 1)

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    empty_columns = int(sum(df[c].isna().all() for c in df.columns))

    duplicate_cells = duplicate_rows * max(cols, 1)

    quality_score = max(
        0,
        min(
            100,
            100 - (missing_cells + duplicate_cells) / total_cells * 100
        )
    )

    return {
        "Rows": rows,
        "Columns": cols,
        "Missing Cells": missing_cells,
        "Duplicate Rows": duplicate_rows,
        "Entirely Empty Columns": empty_columns,
        "Quality Score (%)": round(quality_score, 2)
    }


def analyze_dataset(file):
    try:
        original = read_dataset(file)
    except Exception as exc:
        return (
            f"Upload error: {exc}",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

    proposed = original.copy()
    issue_rows = []
    imputation_rows = []

    # Missing values
    for col in original.columns:
        missing_count = int(original[col].isna().sum())

        if missing_count > 0:
            imputed, method = predict_missing_column(original, col)
            proposed[col] = imputed

            issue_rows.append({
                "Row": "All",
                "Column": col,
                "Problem": "Missing Values",
                "Count": missing_count,
                "Initial Suggestion": method
            })

            imputation_rows.append({
                "Column": col,
                "Missing Before": missing_count,
                "Imputation Method": method,
                "Missing After Suggestion": int(proposed[col].isna().sum())
            })

    # Duplicate rows
    duplicate_count = int(original.duplicated().sum())

    if duplicate_count > 0:
        issue_rows.append({
            "Row": "Multiple",
            "Column": "All Columns",
            "Problem": "Duplicate Rows",
            "Count": duplicate_count,
            "Initial Suggestion": "Review duplicates; remove or keep them"
        })

    # Empty and constant columns
    for col in original.columns:
        if original[col].isna().all():
            issue_rows.append({
                "Row": "All",
                "Column": col,
                "Problem": "Entirely Empty Column",
                "Count": len(original),
                "Initial Suggestion": "Review manually; no values available to infer"
            })

        non_missing = original[col].dropna()

        if len(non_missing) > 0 and non_missing.nunique() == 1:
            issue_rows.append({
                "Row": "All",
                "Column": col,
                "Problem": "Constant Column",
                "Count": len(non_missing),
                "Initial Suggestion": "Review whether this column is useful"
            })

    # Numeric outliers and negative values
    for col in numeric_columns(original):
        values = pd.to_numeric(original[col], errors="coerce").dropna()

        if len(values) >= 5:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1

            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                count = int(((values < lower) | (values > upper)).sum())

                if count > 0:
                    issue_rows.append({
                        "Row": "Multiple",
                        "Column": col,
                        "Problem": "Potential Outliers (IQR)",
                        "Count": count,
                        "Initial Suggestion": "Review outliers; do not remove automatically"
                    })

        negative_count = int((values < 0).sum())

        if negative_count > 0:
            issue_rows.append({
                "Row": "Multiple",
                "Column": col,
                "Problem": "Negative Values",
                "Count": negative_count,
                "Initial Suggestion": "Check whether negative values are valid"
            })

    # Numeric text, whitespace and high cardinality
    for col in categorical_columns(original):
        text_values = original[col].dropna().astype(str).str.strip()

        if len(text_values) > 0:
            converted = pd.to_numeric(text_values, errors="coerce")

            if converted.notna().mean() >= 0.80:
                issue_rows.append({
                    "Row": "Multiple",
                    "Column": col,
                    "Problem": "Numbers Stored as Text",
                    "Count": int(converted.notna().sum()),
                    "Initial Suggestion": "Review whether this column should be numeric"
                })

        whitespace_count = int(
            original[col].dropna().astype(str).apply(
                lambda v: v != v.strip()
            ).sum()
        )

        if whitespace_count > 0:
            issue_rows.append({
                "Row": "Multiple",
                "Column": col,
                "Problem": "Leading/Trailing Spaces",
                "Count": whitespace_count,
                "Initial Suggestion": "Review and trim unnecessary spaces"
            })

        if len(text_values) >= 10:
            unique_fraction = text_values.nunique() / len(text_values)

            if unique_fraction >= 0.90:
                issue_rows.append({
                    "Row": "Multiple",
                    "Column": col,
                    "Problem": "High Cardinality Text Column",
                    "Count": int(text_values.nunique()),
                    "Initial Suggestion": "Review identifiers or free text"
                })

    issues = pd.DataFrame(
        issue_rows,
        columns=["Row", "Column", "Problem", "Count", "Initial Suggestion"]
    )

    imputation_report = pd.DataFrame(
        imputation_rows,
        columns=[
            "Column", "Missing Before", "Imputation Method",
            "Missing After Suggestion"
        ]
    )

    metrics = calculate_quality_metrics(original)

    summary = (
        f"Dataset loaded successfully.\n"
        f"Rows: {metrics['Rows']:,}\n"
        f"Columns: {metrics['Columns']:,}\n"
        f"Missing cells: {metrics['Missing Cells']:,}\n"
        f"Duplicate rows: {metrics['Duplicate Rows']:,}\n"
        f"Potential issues listed: {len(issues):,}\n"
        f"Initial quality score: {metrics['Quality Score (%)']}%\n\n"
        "Human approval is required before suggestions are applied."
    )

    APP["original"] = original.copy()
    APP["proposed"] = proposed.copy()
    APP["approved"] = original.copy()
    APP["issues"] = issues.copy()
    APP["quality_summary"] = pd.DataFrame([metrics])
    APP["imputation_report"] = imputation_report.copy()
    APP["quality_comparison"] = None

    APP["anomaly_report"] = None
    APP["drift_report"] = None
    APP["reference_data"] = None
    APP["current_data"] = None
    APP["quality_comparison_plot"] = None
    APP["anomaly_plot"] = None
    APP["drift_plot"] = None

    return (
        summary,
        issues,
        dataframe_preview(original),
        dataframe_preview(proposed),
        dataframe_preview(APP["approved"])
    )


# ================================================================
# HUMAN-IN-THE-LOOP APPROVAL AND MANUAL OVERRIDE
# ================================================================

def approve_suggestions():
    if APP["proposed"] is None:
        return "Upload and analyze a dataset first.", pd.DataFrame()

    APP["approved"] = APP["proposed"].copy()

    return (
        "Suggested imputation values accepted. You can still edit values "
        "or handle duplicates.",
        dataframe_preview(APP["approved"])
    )


def reject_suggestions():
    if APP["original"] is None:
        return "Upload and analyze a dataset first.", pd.DataFrame()

    APP["approved"] = APP["original"].copy()

    return (
        "Suggestions rejected. Approved dataset matches the original.",
        dataframe_preview(APP["approved"])
    )


def remove_duplicates():
    if APP["approved"] is None:
        return "Upload and analyze a dataset first.", pd.DataFrame()

    before = len(APP["approved"])
    APP["approved"] = APP["approved"].drop_duplicates().reset_index(drop=True)
    removed = before - len(APP["approved"])

    return (
        f"Removed {removed} duplicate row(s).",
        dataframe_preview(APP["approved"])
    )


def keep_duplicates():
    if APP["approved"] is None:
        return "Upload and analyze a dataset first.", pd.DataFrame()

    return (
        f"Kept all rows. Approved dataset contains {len(APP['approved'])} rows.",
        dataframe_preview(APP["approved"])
    )


def apply_human_override(row_number, column_name, new_value):
    if APP["approved"] is None:
        return "Upload and analyze a dataset first.", pd.DataFrame()

    if column_name is None or str(column_name).strip() == "":
        return "Enter a valid column name.", dataframe_preview(APP["approved"])

    if column_name not in APP["approved"].columns:
        return (
            f"Column '{column_name}' was not found.",
            dataframe_preview(APP["approved"])
        )

    try:
        row_number = int(row_number)
    except Exception:
        return "Enter a valid zero-based row number.", dataframe_preview(APP["approved"])

    if row_number < 0 or row_number >= len(APP["approved"]):
        return (
            f"Row number must be between 0 and {len(APP['approved']) - 1}.",
            dataframe_preview(APP["approved"])
        )

    try:
        old_value = APP["approved"].iloc[row_number][column_name]
        series = APP["approved"][column_name]

        if pd.api.types.is_numeric_dtype(series):
            converted_value = (
                np.nan if str(new_value).strip() == ""
                else pd.to_numeric(new_value, errors="raise")
            )
        else:
            converted_value = str(new_value)

        APP["approved"].loc[
            APP["approved"].index[row_number], column_name
        ] = converted_value

        return (
            f"Override applied at row {row_number}, column '{column_name}'. "
            f"Previous: {safe_string(old_value)}; "
            f"New: {safe_string(converted_value)}.",
            dataframe_preview(APP["approved"])
        )

    except Exception as exc:
        return f"Could not apply override: {exc}", dataframe_preview(APP["approved"])


def download_approved_dataset():
    if APP["approved"] is None:
        raise gr.Error("Upload and analyze a dataset first.")

    path = os.path.join(REPORT_DIR, "final_approved_dataset.xlsx")
    APP["approved"].to_excel(path, index=False)
    return path


def download_quality_report():
    if APP["original"] is None:
        raise gr.Error("Upload and analyze a dataset first.")

    path = os.path.join(REPORT_DIR, "data_quality_report.xlsx")

    sheets = {
        "Original Dataset": APP["original"],
        "Proposed Dataset": APP["proposed"],
        "Approved Dataset": APP["approved"],
        "Detected Issues": APP["issues"],
        "Initial Quality": APP["quality_summary"],
        "Imputation Report": APP["imputation_report"]
    }

    if APP["quality_comparison"] is not None:
        sheets["Quality Comparison"] = APP["quality_comparison"]

    APP["quality_report_path"] = make_excel_report(sheets, path)
    return path


# ================================================================
# EXPLAINABLE ANOMALY DETECTION
# AUTOENCODER + ISOLATION FOREST
# ================================================================

def run_anomaly_detection(file, percentile=AE_THRESHOLD_PERCENTILE):
    try:
        df = read_dataset(file)
    except Exception as exc:
        return f"Upload error: {exc}", pd.DataFrame(), None

    features = [
        col for col in numeric_columns(df)
        if not is_identifier_column(col)
    ]

    if not features:
        return (
            "Anomaly detection needs at least one suitable numeric feature.",
            pd.DataFrame(),
            None
        )

    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    features = [
        col for col in X_df.columns
        if X_df[col].notna().any() and X_df[col].nunique(dropna=True) > 1
    ]

    if not features:
        return (
            "No usable varying numeric columns were found.",
            pd.DataFrame(),
            None
        )

    X_df = X_df[features]

    if len(X_df) < 10:
        return (
            "At least 10 rows are recommended for anomaly detection.",
            pd.DataFrame(),
            None
        )

    # Impute and scale numeric data
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    n_rows, n_features = X_scaled.shape
    hidden_units = max(2, min(16, n_features * 2))
    latent_units = max(1, min(8, n_features))

    # Autoencoder architecture
    input_layer = layers.Input(shape=(n_features,))
    encoded = layers.Dense(hidden_units, activation="relu")(input_layer)
    encoded = layers.Dense(latent_units, activation="relu")(encoded)
    decoded = layers.Dense(hidden_units, activation="relu")(encoded)
    output_layer = layers.Dense(n_features, activation="linear")(decoded)

    autoencoder = models.Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")

    early_stop = callbacks.EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )

    autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=50,
        batch_size=min(32, n_rows),
        shuffle=True,
        verbose=0,
        callbacks=[early_stop]
    )

    reconstructed = autoencoder.predict(X_scaled, verbose=0)

    # Feature-level and row-level reconstruction errors
    feature_errors = np.square(X_scaled - reconstructed)
    reconstruction_error = np.mean(feature_errors, axis=1)

    # USER-ADJUSTABLE PERCENTILE; DEFAULT IS 95
    percentile = float(percentile)
    percentile = max(80.0, min(99.9, percentile))

    ae_threshold = float(
        np.percentile(reconstruction_error, percentile)
    )

    ae_flags = reconstruction_error > ae_threshold

    # Explain which features contribute most to the error
    top_feature_explanations = []

    for row_errors in feature_errors:
        top_indices = np.argsort(row_errors)[::-1][:min(3, len(features))]
        explanations = [
            f"{features[i]} (error={row_errors[i]:.4f})"
            for i in top_indices
        ]
        top_feature_explanations.append("; ".join(explanations))

    # Isolation Forest
    contamination = float(np.clip(np.mean(ae_flags), 0.01, 0.20))

    iso_model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    iso_prediction = iso_model.fit_predict(X_scaled)
    iso_flags = iso_prediction == -1
    iso_scores = iso_model.decision_function(X_scaled)

    result = df.copy()
    result["Autoencoder_Error"] = reconstruction_error
    result["Autoencoder_Anomaly"] = ae_flags
    result["Top_Contributing_Features"] = top_feature_explanations
    result["IsolationForest_Score"] = iso_scores
    result["IsolationForest_Anomaly"] = iso_flags
    result["Both_Models_Flagged"] = ae_flags & iso_flags
    result["Either_Model_Flagged"] = ae_flags | iso_flags

    for i, feature_name in enumerate(features):
        safe_name = (
            str(feature_name).replace(" ", "_")
            .replace("/", "_").replace("\\", "_")
        )
        result[f"AE_Error_{safe_name}"] = feature_errors[:, i]

    anomaly_count = int(result["Either_Model_Flagged"].sum())
    both_count = int(result["Both_Models_Flagged"].sum())
    ae_count = int(result["Autoencoder_Anomaly"].sum())
    iso_count = int(result["IsolationForest_Anomaly"].sum())

    summary = (
        f"Anomaly detection completed.\n"
        f"Rows analyzed: {n_rows:,}\n"
        f"Features used: {len(features)}\n"
        f"Autoencoder anomalies: {ae_count:,}\n"
        f"Isolation Forest anomalies: {iso_count:,}\n"
        f"Flagged by both models: {both_count:,}\n"
        f"Flagged by either model: {anomaly_count:,}\n"
        f"Selected threshold percentile: {percentile:g}\n"
        f"Calculated reconstruction-error threshold: {ae_threshold:.6f}\n\n"
        "The threshold is recalculated from the uploaded data each time. "
        "Top_Contributing_Features provides clues for human review, not "
        "proof that a row is incorrect."
    )

    # Plot error distribution and threshold
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(reconstruction_error, bins=30)
    ax.axvline(
        ae_threshold,
        linestyle="--",
        label=f"{percentile:g}th percentile threshold"
    )
    ax.set_title("Autoencoder Reconstruction Error")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Number of Rows")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(REPORT_DIR, "anomaly_error_distribution.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_path = os.path.join(REPORT_DIR, "anomaly_detection_report.xlsx")

    make_excel_report(
        {
            "Anomaly Results": result,
            "Anomaly Summary": pd.DataFrame([{
                "Rows Analyzed": n_rows,
                "Features Used": len(features),
                "Autoencoder Anomalies": ae_count,
                "Isolation Forest Anomalies": iso_count,
                "Both Models Flagged": both_count,
                "Either Model Flagged": anomaly_count,
                "Autoencoder Threshold": ae_threshold,
                "Threshold Percentile": percentile
            }])
        },
        report_path
    )

    APP["anomaly_report"] = result.copy()
    APP["anomaly_plot"] = plot_path
    APP["anomaly_report_path"] = report_path

    return summary, result, plot_path


def download_anomaly_report():
    if APP["anomaly_report"] is None:
        raise gr.Error("Run anomaly detection first.")

    path = APP.get("anomaly_report_path")

    if path is None or not os.path.exists(path):
        path = os.path.join(REPORT_DIR, "anomaly_detection_report.xlsx")
        make_excel_report({"Anomaly Results": APP["anomaly_report"]}, path)
        APP["anomaly_report_path"] = path

    return path


# ================================================================
# DATA DRIFT: PSI + KS + CHI-SQUARE
# ================================================================

def calculate_psi(reference, current, bins=10):
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()

    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    ref_num = pd.to_numeric(ref, errors="coerce")
    cur_num = pd.to_numeric(cur, errors="coerce")

    if ref_num.notna().mean() >= 0.95 and cur_num.notna().mean() >= 0.95:
        ref_num = ref_num.dropna().astype(float)
        cur_num = cur_num.dropna().astype(float)
        combined = pd.concat([ref_num, cur_num], ignore_index=True)

        if combined.nunique() <= 1:
            return 0.0

        edges = np.unique(np.quantile(combined, np.linspace(0, 1, bins + 1)))

        if len(edges) < 2:
            return 0.0

        edges[0], edges[-1] = -np.inf, np.inf

        ref_counts = pd.cut(
            ref_num, bins=edges, include_lowest=True
        ).value_counts(sort=False).to_numpy(dtype=float)

        cur_counts = pd.cut(
            cur_num, bins=edges, include_lowest=True
        ).value_counts(sort=False).to_numpy(dtype=float)

    else:
        ref_cat, cur_cat = ref.astype(str), cur.astype(str)
        categories = sorted(set(ref_cat.unique()) | set(cur_cat.unique()))

        ref_counts = np.array([(ref_cat == c).sum() for c in categories], float)
        cur_counts = np.array([(cur_cat == c).sum() for c in categories], float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def run_drift_analysis(reference_file, current_file):
    try:
        reference = read_dataset(reference_file)
        current = read_dataset(current_file)
    except Exception as exc:
        return f"Upload error: {exc}", pd.DataFrame(), None

    common_columns = [
        c for c in reference.columns if c in current.columns
    ]

    if not common_columns:
        return (
            "The datasets have no common column names.",
            pd.DataFrame(),
            None
        )

    drift_rows = []

    for col in common_columns:
        ref_series, cur_series = reference[col], current[col]
        psi = calculate_psi(ref_series, cur_series)

        if pd.isna(psi):
            drift_level = "Not Tested"
        elif psi < 0.10:
            drift_level = "Low"
        elif psi < 0.25:
            drift_level = "Moderate"
        else:
            drift_level = "High"

        statistical_test = "Not tested"
        statistic, p_value = np.nan, np.nan
        statistical_result = "Not tested"

        ref_num = pd.to_numeric(ref_series, errors="coerce")
        cur_num = pd.to_numeric(cur_series, errors="coerce")

        ref_fraction = ref_num.notna().mean() if len(ref_series.dropna()) else 0
        cur_fraction = cur_num.notna().mean() if len(cur_series.dropna()) else 0

        try:
            if (
                ref_fraction >= 0.95 and cur_fraction >= 0.95
                and ref_num.dropna().nunique() > 1
                and cur_num.dropna().nunique() > 1
                and len(ref_num.dropna()) >= 2
                and len(cur_num.dropna()) >= 2
            ):
                statistic, p_value = ks_2samp(
                    ref_num.dropna(), cur_num.dropna()
                )
                statistical_test = "Kolmogorov-Smirnov (KS)"

            else:
                ref_cat = ref_series.dropna().astype(str)
                cur_cat = cur_series.dropna().astype(str)
                categories = sorted(set(ref_cat.unique()) | set(cur_cat.unique()))

                if len(categories) >= 2 and len(ref_cat) and len(cur_cat):
                    contingency = np.array([
                        [(ref_cat == c).sum() for c in categories],
                        [(cur_cat == c).sum() for c in categories]
                    ], dtype=float)

                    contingency = contingency[:, contingency.sum(axis=0) > 0]

                    if contingency.shape[1] >= 2:
                        statistic, p_value, _, _ = chi2_contingency(contingency)
                        statistical_test = "Chi-Square"

            if not pd.isna(p_value):
                statistical_result = (
                    "Statistically significant" if p_value < 0.05
                    else "Not statistically significant"
                )

        except Exception:
            statistical_result = "Not tested"

        drift_rows.append({
            "Column": col,
            "Reference Rows": len(reference),
            "Current Rows": len(current),
            "Reference Missing": int(ref_series.isna().sum()),
            "Current Missing": int(cur_series.isna().sum()),
            "Reference Unique Values": int(ref_series.nunique(dropna=True)),
            "Current Unique Values": int(cur_series.nunique(dropna=True)),
            "PSI": round(psi, 6) if not pd.isna(psi) else np.nan,
            "Drift Level": drift_level,
            "Statistical Test": statistical_test,
            "Test Statistic": round(float(statistic), 6) if not pd.isna(statistic) else np.nan,
            "P-Value": round(float(p_value), 8) if not pd.isna(p_value) else np.nan,
            "Statistical Result": statistical_result
        })

    drift_report = pd.DataFrame(drift_rows)

    high_count = int((drift_report["Drift Level"] == "High").sum())
    moderate_count = int((drift_report["Drift Level"] == "Moderate").sum())
    significant_count = int(
        (drift_report["Statistical Result"] == "Statistically significant").sum()
    )

    summary = (
        f"Data drift analysis completed.\n"
        f"Reference rows: {len(reference):,}\n"
        f"Current rows: {len(current):,}\n"
        f"Common columns compared: {len(common_columns)}\n"
        f"High PSI drift columns: {high_count}\n"
        f"Moderate PSI drift columns: {moderate_count}\n"
        f"Statistically significant columns (p < 0.05): {significant_count}\n\n"
        "PSI: below 0.10 = low, 0.10 to below 0.25 = moderate, "
        "0.25 or higher = high. KS is used for numeric data and chi-square "
        "for categorical data when suitable. P-values are not adjusted "
        "for multiple comparisons."
    )

    # Color-coded PSI plot: Green = Low, Orange = Moderate, Red = High
    # Low: PSI < 0.10 | Moderate: 0.10 <= PSI < 0.25 | High: PSI >= 0.25
    from matplotlib.patches import Patch

    plot_df = drift_report.dropna(subset=["PSI"]).sort_values("PSI", ascending=True).copy()
    plot_path = os.path.join(REPORT_DIR, "data_drift_plot.png")

    fig_height = max(4.5, min(0.48 * len(plot_df) + 1.8, 18))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    if not plot_df.empty:
        def drift_color(level):
            if level == "Low":
                return "green"
            if level == "Moderate":
                return "orange"
            if level == "High":
                return "red"
            return "gray"

        bar_colors = [drift_color(level) for level in plot_df["Drift Level"]]
        bars = ax.barh(plot_df["Column"].astype(str), plot_df["PSI"], color=bar_colors)

        ax.axvline(0.10, color="orange", linestyle="--", linewidth=1.5,
                   label="Moderate boundary (PSI = 0.10)")
        ax.axvline(0.25, color="red", linestyle="--", linewidth=1.5,
                   label="High boundary (PSI = 0.25)")

        max_psi = float(plot_df["PSI"].max()) if len(plot_df) else 0.0
        label_offset = max(max_psi * 0.015, 0.005)
        for bar, value in zip(bars, plot_df["PSI"]):
            ax.text(value + label_offset, bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}", va="center", fontsize=9)

        legend_items = [
            Patch(facecolor="green", label="Low drift (PSI < 0.10)"),
            Patch(facecolor="orange", label="Moderate drift (0.10–<0.25)"),
            Patch(facecolor="red", label="High drift (PSI ≥ 0.25)")
        ]
        ax.legend(handles=legend_items, loc="lower right", fontsize=8)
        ax.set_xlim(left=0, right=max(max_psi * 1.18, 0.30))
        ax.grid(axis="x", linestyle=":", alpha=0.35)
    else:
        ax.text(0.5, 0.5, "No PSI values available", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_title("Data Drift by Column (PSI) — Color-Coded Severity")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_ylabel("Column")
    fig.tight_layout()

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_path = os.path.join(REPORT_DIR, "data_drift_report.xlsx")

    make_excel_report({
        "Drift Results": drift_report,
        "Reference Dataset": reference,
        "Current Dataset": current
    }, report_path)

    APP["reference_data"] = reference.copy()
    APP["current_data"] = current.copy()
    APP["drift_report"] = drift_report.copy()
    APP["drift_plot"] = plot_path
    APP["drift_report_path"] = report_path

    return summary, drift_report, plot_path


def download_drift_report():
    if APP["drift_report"] is None:
        raise gr.Error("Run data drift analysis first.")

    path = APP.get("drift_report_path")

    if path is None or not os.path.exists(path):
        path = os.path.join(REPORT_DIR, "data_drift_report.xlsx")
        make_excel_report({"Drift Results": APP["drift_report"]}, path)
        APP["drift_report_path"] = path

    return path


# ================================================================
# BEFORE-AND-AFTER QUALITY COMPARISON
# ================================================================

def compare_quality_before_after():
    if APP["original"] is None or APP["approved"] is None:
        return "Upload and analyze a dataset first.", pd.DataFrame(), None

    before = calculate_quality_metrics(APP["original"])
    after = calculate_quality_metrics(APP["approved"])

    metric_names = [
        "Rows", "Columns", "Missing Cells", "Duplicate Rows",
        "Entirely Empty Columns", "Quality Score (%)"
    ]

    comparison_rows = []

    for metric in metric_names:
        b, a = before[metric], after[metric]
        change = round(float(a) - float(b), 4)

        if metric == "Quality Score (%)":
            interpretation = (
                "Quality score improved" if change > 0
                else "Quality score decreased" if change < 0
                else "No change in quality score"
            )
        elif metric in ["Missing Cells", "Duplicate Rows", "Entirely Empty Columns"]:
            interpretation = (
                "Reduced" if change < 0
                else "Increased" if change > 0
                else "No change"
            )
        else:
            interpretation = "Informational"

        comparison_rows.append({
            "Metric": metric,
            "Before": b,
            "After": a,
            "Change (After - Before)": change,
            "Interpretation": interpretation
        })

    comparison_df = pd.DataFrame(comparison_rows)
    APP["quality_comparison"] = comparison_df.copy()

    # Draw counts and quality score separately because the score is a
    # percentage while the other metrics are counts. Always annotate bars,
    # including zero-height bars, so an After value of 0 is still visible.
    chart_metrics = ["Missing Cells", "Duplicate Rows", "Entirely Empty Columns"]
    before_counts = [int(before[m]) for m in chart_metrics]
    after_counts = [int(after[m]) for m in chart_metrics]
    x = np.arange(len(chart_metrics))
    width = 0.34

    fig, (ax, score_ax) = plt.subplots(
        2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    bars_before = ax.bar(x - width / 2, before_counts, width, label="Before", color="#2878B5")
    bars_after = ax.bar(x + width / 2, after_counts, width, label="After", color="#F28E2B")
    ax.set_xticks(x)
    ax.set_xticklabels(chart_metrics, rotation=10, ha="right")
    ax.set_title("Data Quality: Before vs After Human Approval")
    ax.set_ylabel("Number of issues")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    max_count = max(before_counts + after_counts + [0])
    ax.set_ylim(0, max(1, max_count * 1.25))

    for bars, values in ((bars_before, before_counts), (bars_after, after_counts)):
        for bar, value in zip(bars, values):
            # Put zero labels slightly above the baseline because zero-height bars
            # cannot display their own labels reliably.
            y = bar.get_height()
            ax.annotate(
                str(value),
                xy=(bar.get_x() + bar.get_width() / 2, y),
                xytext=(0, 4 if value > 0 else 2),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

    score_before = float(before["Quality Score (%)"])
    score_after = float(after["Quality Score (%)"])
    score_bars = score_ax.bar(
        ["Before", "After"], [score_before, score_after],
        color=["#2878B5", "#F28E2B"], width=0.5
    )
    score_ax.set_title("Overall Quality Score")
    score_ax.set_ylabel("Score (%)")
    score_ax.set_ylim(0, 110)
    score_ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(score_bars, [score_before, score_after]):
        score_ax.annotate(
            f"{value:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    fig.tight_layout()

    plot_path = os.path.join(REPORT_DIR, "quality_before_after.png")
    fig.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    APP["quality_comparison_plot"] = plot_path

    summary = (
        "Before-and-after comparison completed.\n"
        f"Quality score before: {before['Quality Score (%)']}%\n"
        f"Quality score after: {after['Quality Score (%)']}%\n"
        f"Score change: {after['Quality Score (%)'] - before['Quality Score (%)']:+.2f} percentage points.\n\n"
        "The quality score is a heuristic based on missing cells and "
        "duplicate rows, not a universal data quality standard."
    )

    return summary, comparison_df, plot_path


# ================================================================
# COMBINED EXCEL REPORT
# ================================================================

def download_combined_report():
    if APP["original"] is None:
        raise gr.Error("Upload and analyze a dataset first.")

    path = os.path.join(REPORT_DIR, "combined_monitoring_report.xlsx")

    sheets = {
        "Original Dataset": APP["original"],
        "Proposed Dataset": APP["proposed"],
        "Approved Dataset": APP["approved"],
        "Quality Issues": APP["issues"],
        "Initial Quality": APP["quality_summary"],
        "Imputation Report": APP["imputation_report"]
    }

    if APP.get("quality_comparison") is not None:
        sheets["Quality Comparison"] = APP["quality_comparison"]

    if APP.get("anomaly_report") is not None:
        sheets["Anomaly Results"] = APP["anomaly_report"]

    if APP.get("drift_report") is not None:
        sheets["Drift Results"] = APP["drift_report"]

    make_excel_report(sheets, path)
    APP["combined_report_path"] = path
    return path


# ================================================================
# GRADIO DASHBOARD
# ================================================================

custom_css = """
.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
}
"""

with gr.Blocks(
    title="Intelligent Data Quality Monitoring Platform",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=custom_css
) as dashboard:

    gr.Markdown("""
# Intelligent Data Quality Monitoring Platform

Upload datasets, inspect quality problems, review suggested changes,
detect anomalies, compare data distributions, and export reports.

**Human approval remains part of the workflow:** suggested values are not
applied to the approved dataset until you accept them.
""")

    with gr.Tabs():

        # --------------------------------------------------------
        # TAB 1: DATA QUALITY AND HUMAN APPROVAL
        # --------------------------------------------------------

        with gr.Tab("Data Quality & Human Approval"):

            gr.Markdown("### 1. Upload and Analyze Dataset")

            dataset_file = gr.File(
                label="Upload CSV / XLS / XLSX Dataset",
                file_types=[".csv", ".xls", ".xlsx"],
                type="filepath"
            )

            analyze_btn = gr.Button("Analyze Dataset", variant="primary")

            quality_summary_output = gr.Textbox(label="Dataset Summary", lines=8)
            issues_output = gr.Dataframe(label="Detected Data Quality Issues", interactive=False)

            with gr.Row():
                original_preview = gr.Dataframe(label="Original Dataset Preview", interactive=False)
                proposed_preview = gr.Dataframe(label="Proposed Dataset Preview", interactive=False)

            approved_preview = gr.Dataframe(label="Approved Dataset Preview", interactive=False)

            analyze_btn.click(
                fn=analyze_dataset,
                inputs=[dataset_file],
                outputs=[
                    quality_summary_output, issues_output,
                    original_preview, proposed_preview, approved_preview
                ]
            )

            gr.Markdown("### 2. Human Approval of Suggested Values")

            with gr.Row():
                accept_btn = gr.Button("Accept Suggested Values", variant="primary")
                reject_btn = gr.Button("Reject Suggested Values")

            approval_message = gr.Textbox(label="Approval Status", lines=2)

            accept_btn.click(
                fn=approve_suggestions,
                inputs=[],
                outputs=[approval_message, approved_preview]
            )

            reject_btn.click(
                fn=reject_suggestions,
                inputs=[],
                outputs=[approval_message, approved_preview]
            )

            with gr.Row():
                remove_duplicates_btn = gr.Button("Human Decision: Remove Duplicates")
                keep_duplicates_btn = gr.Button("Human Decision: Keep Duplicates")

            duplicate_message = gr.Textbox(label="Duplicate Handling Status", lines=2)

            remove_duplicates_btn.click(
                fn=remove_duplicates,
                inputs=[],
                outputs=[duplicate_message, approved_preview]
            )

            keep_duplicates_btn.click(
                fn=keep_duplicates,
                inputs=[],
                outputs=[duplicate_message, approved_preview]
            )

            gr.Markdown("""
### 3. Manual Human Override

Enter the zero-based row number and exact column name.
Your replacement value is applied to the approved dataset.
""")

            with gr.Row():
                override_row = gr.Number(label="Row Number (starts at 0)", value=0, precision=0)
                override_column = gr.Textbox(label="Exact Column Name", placeholder="Example: Age")
                override_value = gr.Textbox(label="New Value", placeholder="Enter replacement value")

            override_btn = gr.Button("Apply Human Override", variant="primary")
            override_message = gr.Textbox(label="Override Status", lines=2)

            override_btn.click(
                fn=apply_human_override,
                inputs=[override_row, override_column, override_value],
                outputs=[override_message, approved_preview]
            )

            gr.Markdown("### 4. Download Quality Results")

            with gr.Row():
                quality_report_btn = gr.Button("Generate Quality Excel Report")
                approved_dataset_btn = gr.Button("Download Approved Dataset")

            quality_report_file = gr.File(label="Quality Report Download")
            approved_dataset_file = gr.File(label="Approved Dataset Download")

            quality_report_btn.click(
                fn=download_quality_report,
                inputs=[],
                outputs=quality_report_file
            )

            approved_dataset_btn.click(
                fn=download_approved_dataset,
                inputs=[],
                outputs=approved_dataset_file
            )

        # --------------------------------------------------------
        # TAB 2: ANOMALY DETECTION
        # --------------------------------------------------------

        with gr.Tab("Anomaly Detection"):

            gr.Markdown("""
## Explainable Anomaly Detection

**Autoencoder:** learns patterns and flags rows with large reconstruction errors.

**Isolation Forest:** detects rows that are unusual relative to other rows.

Feature explanations identify columns with the largest reconstruction errors.
Review flagged rows before taking action.
""")

            anomaly_file = gr.File(
                label="Upload Dataset for Anomaly Detection",
                file_types=[".csv", ".xls", ".xlsx"],
                type="filepath"
            )

            gr.Markdown("""
### Autoencoder Error Threshold Percentile

The default is the 95th percentile. Adjust the slider when needed.

- Lower percentile: generally more rows flagged.
- Higher percentile: generally fewer rows flagged.
- The numerical error threshold is recalculated automatically.
""")

            anomaly_percentile = gr.Slider(
                minimum=80,
                maximum=99.9,
                value=AE_THRESHOLD_PERCENTILE,
                step=0.5,
                label="Autoencoder Error Threshold Percentile"
            )

            anomaly_btn = gr.Button("Run Anomaly Detection", variant="primary")

            anomaly_summary_output = gr.Textbox(
                label="Anomaly Detection Summary",
                lines=9
            )

            anomaly_results_output = gr.Dataframe(
                label="Anomaly Results and Feature Explanations",
                interactive=False
            )

            anomaly_plot_output = gr.Image(
                label="Autoencoder Error Distribution",
                type="filepath"
            )

            # Pass the user's selected percentile into the function
            anomaly_btn.click(
                fn=run_anomaly_detection,
                inputs=[anomaly_file, anomaly_percentile],
                outputs=[
                    anomaly_summary_output,
                    anomaly_results_output,
                    anomaly_plot_output
                ]
            )

            anomaly_download_btn = gr.Button("Download Anomaly Excel Report")
            anomaly_download_file = gr.File(label="Anomaly Report Download")

            anomaly_download_btn.click(
                fn=download_anomaly_report,
                inputs=[],
                outputs=anomaly_download_file
            )

        # --------------------------------------------------------
        # TAB 3: DATA DRIFT
        # --------------------------------------------------------

        with gr.Tab("Data Drift Analysis"):

            gr.Markdown("""
## Multiple Data Drift Detection Methods

Upload two datasets with matching column names.

- **Reference dataset:** baseline or earlier data.
- **Current dataset:** newer data to compare.
- **PSI:** measures distribution changes.
- **KS test:** compares numerical distributions.
- **Chi-square test:** compares categorical distributions.

Statistical significance does not automatically mean the data are bad.
""")

            with gr.Row():
                reference_file = gr.File(
                    label="Reference Dataset",
                    file_types=[".csv", ".xls", ".xlsx"],
                    type="filepath"
                )
                current_file = gr.File(
                    label="Current Dataset",
                    file_types=[".csv", ".xls", ".xlsx"],
                    type="filepath"
                )

            drift_btn = gr.Button("Run Multiple Drift Analysis", variant="primary")
            drift_summary_output = gr.Textbox(label="Drift Summary", lines=9)

            drift_results_output = gr.Dataframe(
                label="PSI and Statistical Test Results",
                interactive=False
            )

            drift_plot_output = gr.Image(label="PSI Drift Plot", type="filepath")

            drift_btn.click(
                fn=run_drift_analysis,
                inputs=[reference_file, current_file],
                outputs=[drift_summary_output, drift_results_output, drift_plot_output]
            )

            drift_download_btn = gr.Button("Download Drift Excel Report")
            drift_download_file = gr.File(label="Drift Report Download")

            drift_download_btn.click(
                fn=download_drift_report,
                inputs=[],
                outputs=drift_download_file
            )

        # --------------------------------------------------------
        # TAB 4: QUALITY COMPARISON
        # --------------------------------------------------------

        with gr.Tab("Quality Comparison"):

            gr.Markdown("""
## Before-and-After Quality Comparison

Compare the original dataset with the currently approved dataset.
Accept or reject suggestions, handle duplicates, or apply manual overrides
in the Data Quality tab first, then click Compare again to refresh the chart.
The chart labels include zero values, and the quality score is shown separately.
""")

            compare_btn = gr.Button(
                "Compare Original vs Approved Dataset",
                variant="primary"
            )

            comparison_summary_output = gr.Textbox(label="Comparison Summary", lines=6)

            comparison_table_output = gr.Dataframe(
                label="Before-and-After Metrics",
                interactive=False
            )

            comparison_plot_output = gr.Image(
                label="Quality Comparison Chart",
                type="filepath"
            )

            compare_btn.click(
                fn=compare_quality_before_after,
                inputs=[],
                outputs=[
                    comparison_summary_output,
                    comparison_table_output,
                    comparison_plot_output
                ]
            )

        # --------------------------------------------------------
        # TAB 5: COMBINED REPORT
        # --------------------------------------------------------

        with gr.Tab("Combined Report"):

            gr.Markdown("""
## Combined Monitoring Report

Generate an Excel workbook containing the datasets and analyses run in
this session. Run the relevant analysis tabs first to include their results.
""")

            combined_report_btn = gr.Button(
                "Generate Combined Excel Report",
                variant="primary"
            )

            combined_report_file = gr.File(label="Combined Report Download")

            combined_report_btn.click(
                fn=download_combined_report,
                inputs=[],
                outputs=combined_report_file
            )

    gr.Markdown("""
---
**Important:** Anomaly and drift findings are signals for investigation,
not automatic proof of errors. Human review is recommended before modifying
or using a production dataset.
""")


# ================================================================
# LAUNCH DASHBOARD
# ================================================================

if __name__ == "__main__":
    dashboard.launch(share=True, debug=False)
