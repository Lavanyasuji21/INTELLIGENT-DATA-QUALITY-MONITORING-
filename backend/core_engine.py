import os
import sys
import warnings

# Ensure writable directories for Matplotlib and Keras BEFORE importing them
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
KERAS_DIR = os.path.join(CACHE_DIR, "keras")
MPL_DIR = os.path.join(CACHE_DIR, "matplotlib")
os.makedirs(KERAS_DIR, exist_ok=True)
os.makedirs(MPL_DIR, exist_ok=True)
os.environ["KERAS_HOME"] = KERAS_DIR
os.environ["MPLCONFIGDIR"] = MPL_DIR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for web server
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import ks_2samp, chi2_contingency
from tensorflow.keras import layers, models, callbacks
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

AE_THRESHOLD_PERCENTILE = 95


class DataQualityEngine:
    """
    Session-isolated backend engine preserving all original machine learning algorithms,
    statistical metrics, quality calculations, and Excel report generators exactly as provided.
    """

    def __init__(self, session_id: str, base_report_dir: str = None):
        self.session_id = session_id
        if base_report_dir is None:
            self.report_dir = os.path.join(BASE_DIR, "reports", session_id)
        else:
            self.report_dir = os.path.join(base_report_dir, session_id)
        os.makedirs(self.report_dir, exist_ok=True)

        self.APP = {
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
    # GENERAL UTILITIES (PRESERVED EXACTLY)
    # ================================================================

    @staticmethod
    def read_dataset(file):
        if file is None:
            raise ValueError("Please upload a dataset first.")

        if isinstance(file, pd.DataFrame):
            if file.empty:
                raise ValueError("The uploaded dataset is empty.")
            df = file.copy()
            df.columns = [
                str(c).strip() if str(c).strip() else f"Unnamed_{i}"
                for i, c in enumerate(df.columns)
            ]
            return df

        path = file.name if hasattr(file, "name") else str(file)
        extension = os.path.splitext(path)[1].lower()

        # If file is a Streamlit UploadedFile or file-like buffer
        if hasattr(file, "read") and not (isinstance(file, str) or (isinstance(path, str) and os.path.exists(path))):
            if hasattr(file, "seek"):
                file.seek(0)
            if extension == ".csv":
                try:
                    df = pd.read_csv(file)
                except UnicodeDecodeError:
                    if hasattr(file, "seek"):
                        file.seek(0)
                    df = pd.read_csv(file, encoding="latin-1")
            elif extension in [".xlsx", ".xls"]:
                df = pd.read_excel(file)
            else:
                raise ValueError("Supported formats are CSV, XLS and XLSX.")
        else:
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

    @staticmethod
    def safe_string(value):
        if pd.isna(value):
            return ""
        return str(value)

    @staticmethod
    def numeric_columns(df):
        return list(df.select_dtypes(include=np.number).columns)

    @staticmethod
    def categorical_columns(df):
        return list(df.select_dtypes(exclude=np.number).columns)

    @staticmethod
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

    @staticmethod
    def make_excel_report(sheets, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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

    @staticmethod
    def dataframe_preview(df, rows=15):
        if df is None:
            return pd.DataFrame()
        return df.head(rows).copy()

    # ================================================================
    # RANDOM FOREST IMPUTATION (PRESERVED EXACTLY)
    # ================================================================

    @staticmethod
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

    def predict_missing_column(self, df, target_column):
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
            X_all = self.prepare_predictors(df, target_column)

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
    # DATA QUALITY ANALYSIS (PRESERVED EXACTLY)
    # ================================================================

    @staticmethod
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

    def analyze_dataset(self, file):
        try:
            original = self.read_dataset(file)
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
                imputed, method = self.predict_missing_column(original, col)
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
        for col in self.numeric_columns(original):
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
        for col in self.categorical_columns(original):
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

        metrics = self.calculate_quality_metrics(original)

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

        self.APP["original"] = original.copy()
        self.APP["proposed"] = proposed.copy()
        self.APP["approved"] = original.copy()
        self.APP["issues"] = issues.copy()
        self.APP["quality_summary"] = pd.DataFrame([metrics])
        self.APP["imputation_report"] = imputation_report.copy()
        self.APP["quality_comparison"] = None

        self.APP["anomaly_report"] = None
        self.APP["drift_report"] = None
        self.APP["reference_data"] = None
        self.APP["current_data"] = None
        self.APP["quality_comparison_plot"] = None
        self.APP["anomaly_plot"] = None
        self.APP["drift_plot"] = None

        return (
            summary,
            issues,
            self.dataframe_preview(original),
            self.dataframe_preview(proposed),
            self.dataframe_preview(self.APP["approved"])
        )

    # ================================================================
    # HUMAN-IN-THE-LOOP APPROVAL AND MANUAL OVERRIDE (PRESERVED EXACTLY)
    # ================================================================

    def approve_suggestions(self):
        if self.APP["proposed"] is None:
            return "Upload and analyze a dataset first.", pd.DataFrame()

        self.APP["approved"] = self.APP["proposed"].copy()

        return (
            "Suggested imputation values accepted. You can still edit values "
            "or handle duplicates.",
            self.dataframe_preview(self.APP["approved"])
        )

    def reject_suggestions(self):
        if self.APP["original"] is None:
            return "Upload and analyze a dataset first.", pd.DataFrame()

        self.APP["approved"] = self.APP["original"].copy()

        return (
            "Suggestions rejected. Approved dataset matches the original.",
            self.dataframe_preview(self.APP["approved"])
        )

    def remove_duplicates(self):
        if self.APP["approved"] is None:
            return "Upload and analyze a dataset first.", pd.DataFrame()

        before = len(self.APP["approved"])
        self.APP["approved"] = self.APP["approved"].drop_duplicates().reset_index(drop=True)
        removed = before - len(self.APP["approved"])

        return (
            f"Removed {removed} duplicate row(s).",
            self.dataframe_preview(self.APP["approved"])
        )

    def keep_duplicates(self):
        if self.APP["approved"] is None:
            return "Upload and analyze a dataset first.", pd.DataFrame()

        return (
            f"Kept all rows. Approved dataset contains {len(self.APP['approved'])} rows.",
            self.dataframe_preview(self.APP["approved"])
        )

    def apply_human_override(self, row_number, column_name, new_value):
        if self.APP["approved"] is None:
            return "Upload and analyze a dataset first.", pd.DataFrame()

        if column_name is None or str(column_name).strip() == "":
            return "Enter a valid column name.", self.dataframe_preview(self.APP["approved"])

        if column_name not in self.APP["approved"].columns:
            return (
                f"Column '{column_name}' was not found.",
                self.dataframe_preview(self.APP["approved"])
            )

        try:
            row_number = int(row_number)
        except Exception:
            return "Enter a valid zero-based row number.", self.dataframe_preview(self.APP["approved"])

        if row_number < 0 or row_number >= len(self.APP["approved"]):
            return (
                f"Row number must be between 0 and {len(self.APP['approved']) - 1}.",
                self.dataframe_preview(self.APP["approved"])
            )

        try:
            old_value = self.APP["approved"].iloc[row_number][column_name]
            series = self.APP["approved"][column_name]

            if pd.api.types.is_numeric_dtype(series):
                converted_value = (
                    np.nan if str(new_value).strip() == ""
                    else pd.to_numeric(new_value, errors="raise")
                )
            else:
                converted_value = str(new_value)

            self.APP["approved"].loc[
                self.APP["approved"].index[row_number], column_name
            ] = converted_value

            return (
                f"Override applied at row {row_number}, column '{column_name}'. "
                f"Previous: {self.safe_string(old_value)}; "
                f"New: {self.safe_string(converted_value)}.",
                self.dataframe_preview(self.APP["approved"])
            )

        except Exception as exc:
            return f"Could not apply override: {exc}", self.dataframe_preview(self.APP["approved"])

    def download_approved_dataset(self):
        if self.APP["approved"] is None:
            raise ValueError("Upload and analyze a dataset first.")

        path = os.path.join(self.report_dir, "final_approved_dataset.xlsx")
        self.APP["approved"].to_excel(path, index=False)
        return path

    def download_quality_report(self):
        if self.APP["original"] is None:
            raise ValueError("Upload and analyze a dataset first.")

        path = os.path.join(self.report_dir, "data_quality_report.xlsx")

        sheets = {
            "Original Dataset": self.APP["original"],
            "Proposed Dataset": self.APP["proposed"],
            "Approved Dataset": self.APP["approved"],
            "Detected Issues": self.APP["issues"],
            "Initial Quality": self.APP["quality_summary"],
            "Imputation Report": self.APP["imputation_report"]
        }

        if self.APP["quality_comparison"] is not None:
            sheets["Quality Comparison"] = self.APP["quality_comparison"]

        self.APP["quality_report_path"] = self.make_excel_report(sheets, path)
        return path

    # ================================================================
    # EXPLAINABLE ANOMALY DETECTION (PRESERVED EXACTLY)
    # ================================================================

    def run_anomaly_detection(self, file, percentile=AE_THRESHOLD_PERCENTILE):
        try:
            df = self.read_dataset(file)
        except Exception as exc:
            return f"Upload error: {exc}", pd.DataFrame(), None

        features = [
            col for col in self.numeric_columns(df)
            if not self.is_identifier_column(col)
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
        ax.hist(reconstruction_error, bins=30, color="#0d9488", edgecolor="white", alpha=0.85)
        ax.axvline(
            ae_threshold,
            color="#ef4444",
            linestyle="--",
            linewidth=2,
            label=f"{percentile:g}th percentile threshold ({ae_threshold:.4f})"
        )
        ax.set_title("Autoencoder Reconstruction Error", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Reconstruction Error", fontsize=10)
        ax.set_ylabel("Number of Rows", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
        fig.tight_layout()

        plot_path = os.path.join(self.report_dir, "anomaly_error_distribution.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        report_path = os.path.join(self.report_dir, "anomaly_detection_report.xlsx")

        self.make_excel_report(
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

        self.APP["anomaly_report"] = result.copy()
        self.APP["anomaly_plot"] = plot_path
        self.APP["anomaly_report_path"] = report_path

        return summary, result, plot_path

    def download_anomaly_report(self):
        if self.APP["anomaly_report"] is None:
            raise ValueError("Run anomaly detection first.")

        path = self.APP.get("anomaly_report_path")

        if path is None or not os.path.exists(path):
            path = os.path.join(self.report_dir, "anomaly_detection_report.xlsx")
            self.make_excel_report({"Anomaly Results": self.APP["anomaly_report"]}, path)
            self.APP["anomaly_report_path"] = path

        return path

    # ================================================================
    # DATA DRIFT: PSI + KS + CHI-SQUARE (PRESERVED EXACTLY)
    # ================================================================

    @staticmethod
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

    def run_drift_analysis(self, reference_file, current_file):
        try:
            reference = self.read_dataset(reference_file)
            current = self.read_dataset(current_file)
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
            psi = self.calculate_psi(ref_series, cur_series)

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

        plot_df = drift_report.dropna(subset=["PSI"]).sort_values("PSI", ascending=True).copy()
        plot_path = os.path.join(self.report_dir, "data_drift_plot.png")

        fig_height = max(4.5, min(0.48 * len(plot_df) + 1.8, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))

        if not plot_df.empty:
            def drift_color(level):
                if level == "Low":
                    return "#10b981"  # Emerald green
                if level == "Moderate":
                    return "#f59e0b"  # Amber orange
                if level == "High":
                    return "#ef4444"  # Red
                return "#94a3b8"

            bar_colors = [drift_color(level) for level in plot_df["Drift Level"]]
            bars = ax.barh(plot_df["Column"].astype(str), plot_df["PSI"], color=bar_colors, height=0.6)

            ax.axvline(0.10, color="#f59e0b", linestyle="--", linewidth=1.5,
                       label="Moderate boundary (PSI = 0.10)")
            ax.axvline(0.25, color="#ef4444", linestyle="--", linewidth=1.5,
                       label="High boundary (PSI = 0.25)")

            max_psi = float(plot_df["PSI"].max()) if len(plot_df) else 0.0
            label_offset = max(max_psi * 0.015, 0.005)
            for bar, value in zip(bars, plot_df["PSI"]):
                ax.text(value + label_offset, bar.get_y() + bar.get_height() / 2,
                        f"{value:.3f}", va="center", fontsize=9, fontweight="bold", color="#1e293b")

            legend_items = [
                Patch(facecolor="#10b981", label="Low drift (PSI < 0.10)"),
                Patch(facecolor="#f59e0b", label="Moderate drift (0.10–<0.25)"),
                Patch(facecolor="#ef4444", label="High drift (PSI ≥ 0.25)")
            ]
            ax.legend(handles=legend_items, loc="lower right", fontsize=8, frameon=True, facecolor="white")
            ax.set_xlim(left=0, right=max(max_psi * 1.18, 0.30))
            ax.grid(axis="x", linestyle=":", alpha=0.35)
        else:
            ax.text(0.5, 0.5, "No PSI values available", ha="center", va="center",
                    transform=ax.transAxes)

        ax.set_title("Data Drift by Column (PSI) — Color-Coded Severity", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Population Stability Index (PSI)", fontsize=10)
        ax.set_ylabel("Column", fontsize=10)
        fig.tight_layout()

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        report_path = os.path.join(self.report_dir, "data_drift_report.xlsx")

        self.make_excel_report({
            "Drift Results": drift_report,
            "Reference Dataset": reference,
            "Current Dataset": current
        }, report_path)

        self.APP["reference_data"] = reference.copy()
        self.APP["current_data"] = current.copy()
        self.APP["drift_report"] = drift_report.copy()
        self.APP["drift_plot"] = plot_path
        self.APP["drift_report_path"] = report_path

        return summary, drift_report, plot_path

    def download_drift_report(self):
        if self.APP["drift_report"] is None:
            raise ValueError("Run data drift analysis first.")

        path = self.APP.get("drift_report_path")

        if path is None or not os.path.exists(path):
            path = os.path.join(self.report_dir, "data_drift_report.xlsx")
            self.make_excel_report({"Drift Results": self.APP["drift_report"]}, path)
            self.APP["drift_report_path"] = path

        return path

    # ================================================================
    # BEFORE-AND-AFTER QUALITY COMPARISON (PRESERVED EXACTLY)
    # ================================================================

    def compare_quality_before_after(self):
        if self.APP["original"] is None or self.APP["approved"] is None:
            return "Upload and analyze a dataset first.", pd.DataFrame(), None

        before = self.calculate_quality_metrics(self.APP["original"])
        after = self.calculate_quality_metrics(self.APP["approved"])

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
        self.APP["quality_comparison"] = comparison_df.copy()

        chart_metrics = ["Missing Cells", "Duplicate Rows", "Entirely Empty Columns"]
        before_counts = [int(before[m]) for m in chart_metrics]
        after_counts = [int(after[m]) for m in chart_metrics]
        x = np.arange(len(chart_metrics))
        width = 0.34

        fig, (ax, score_ax) = plt.subplots(
            2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]}
        )

        bars_before = ax.bar(x - width / 2, before_counts, width, label="Before", color="#0284c7")
        bars_after = ax.bar(x + width / 2, after_counts, width, label="After", color="#0d9488")
        ax.set_xticks(x)
        ax.set_xticklabels(chart_metrics, rotation=10, ha="right", fontsize=10)
        ax.set_title("Data Quality: Before vs After Human Approval", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Number of issues", fontsize=10)
        ax.legend(loc="best", frameon=True, facecolor="white")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        max_count = max(before_counts + after_counts + [0])
        ax.set_ylim(0, max(1, max_count * 1.25))

        for bars, values in ((bars_before, before_counts), (bars_after, after_counts)):
            for bar, value in zip(bars, values):
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
            color=["#0284c7", "#0d9488"], width=0.45
        )
        score_ax.set_title("Overall Quality Score", fontsize=12, fontweight="bold", pad=8)
        score_ax.set_ylabel("Score (%)", fontsize=10)
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

        plot_path = os.path.join(self.report_dir, "quality_before_after.png")
        fig.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        self.APP["quality_comparison_plot"] = plot_path

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
    # COMBINED EXCEL REPORT (PRESERVED EXACTLY)
    # ================================================================

    def download_combined_report(self):
        if self.APP["original"] is None:
            raise ValueError("Upload and analyze a dataset first.")

        path = os.path.join(self.report_dir, "combined_monitoring_report.xlsx")

        sheets = {
            "Original Dataset": self.APP["original"],
            "Proposed Dataset": self.APP["proposed"],
            "Approved Dataset": self.APP["approved"],
            "Quality Issues": self.APP["issues"],
            "Initial Quality": self.APP["quality_summary"],
            "Imputation Report": self.APP["imputation_report"]
        }

        if self.APP.get("quality_comparison") is not None:
            sheets["Quality Comparison"] = self.APP["quality_comparison"]

        if self.APP.get("anomaly_report") is not None:
            sheets["Anomaly Results"] = self.APP["anomaly_report"]

        if self.APP.get("drift_report") is not None:
            sheets["Drift Results"] = self.APP["drift_report"]

        self.make_excel_report(sheets, path)
        self.APP["combined_report_path"] = path
        return path


# In-memory session manager for multi-user isolation
_sessions = {}

def get_session(session_id: str) -> DataQualityEngine:
    if not session_id:
        session_id = "default"
    if session_id not in _sessions:
        _sessions[session_id] = DataQualityEngine(session_id)
    return _sessions[session_id]

def reset_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return get_session(session_id)
