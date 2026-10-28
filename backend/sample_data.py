import os
import numpy as np
import pandas as pd

def generate_sample_quality_dataset():
    """
    Generates a realistic test dataset containing missing values, duplicates,
    outliers, negative numbers, whitespace, numbers-as-text, and constant columns
    to demonstrate all features of the Intelligent Data Quality Monitoring Platform.
    """
    np.random.seed(42)
    n = 60

    ages = np.random.randint(22, 65, size=n).astype(float)
    incomes = np.random.normal(55000, 15000, size=n).round(2)
    credit_scores = np.random.randint(580, 820, size=n).astype(object)
    departments = np.random.choice(["Engineering", "Marketing", "Sales", "Finance", "HR"], size=n).astype(object)
    experience = (ages - 21) * 0.75 + np.random.normal(0, 2, size=n)
    experience = np.maximum(0, experience).round(1)

    df = pd.DataFrame({
        "TransactionID": [f"TX-{1000 + i}" for i in range(n)],
        "Age": ages,
        "Income": incomes,
        "CreditScore": credit_scores,
        "Department": departments,
        "ExperienceYears": experience,
        "Status": ["Active"] * n  # Constant column
    })

    # 1. Introduce Missing Values (for Random Forest Imputation)
    df.loc[[3, 8, 14, 25, 42], "Age"] = np.nan
    df.loc[[5, 12, 19, 33, 50], "Income"] = np.nan
    df.loc[[7, 21, 35], "Department"] = np.nan

    # 2. Outliers and Negative Values
    df.loc[10, "Age"] = -4.0   # Negative value
    df.loc[15, "Income"] = 350000.0  # Positive outlier
    df.loc[22, "ExperienceYears"] = 85.0  # Extreme outlier

    # 3. Numbers as text and whitespace
    df["CreditScore"] = df["CreditScore"].astype(str)
    df.loc[46:50, "CreditScore"] = "Pending"  # Non-NA string so read_csv treats column as text
    df.loc[[2, 11, 29], "Department"] = "  Marketing  "  # Whitespace

    # 4. Duplicate rows (append duplicates of rows 0 and 1)
    duplicates = df.iloc[[0, 1]].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def generate_sample_drift_datasets():
    """
    Generates compatible reference and current datasets with calibrated
    Low, Moderate, and High PSI drift and statistical test results.
    """
    np.random.seed(42)
    n_ref = 200
    n_cur = 200

    # Stable column -> Low PSI (< 0.10)
    ref_stable = np.random.normal(100, 15, n_ref)
    cur_stable = np.random.normal(101, 15, n_cur)

    # Moderate drift column -> Moderate PSI (0.10 - 0.25)
    ref_mod = np.random.normal(50, 10, n_ref)
    cur_mod = np.random.normal(54, 11, n_cur)

    # High drift column -> High PSI (>= 0.25)
    ref_high = np.random.exponential(scale=20, size=n_ref)
    cur_high = np.random.exponential(scale=45, size=n_cur)

    # Categorical column with drift
    categories = ["Tier-A", "Tier-B", "Tier-C", "Tier-D"]
    ref_cat = np.random.choice(categories, p=[0.4, 0.3, 0.2, 0.1], size=n_ref)
    cur_cat = np.random.choice(categories, p=[0.1, 0.2, 0.3, 0.4], size=n_cur)

    reference_df = pd.DataFrame({
        "StableMetric": np.round(ref_stable, 2),
        "ModerateMetric": np.round(ref_mod, 2),
        "HighShiftMetric": np.round(ref_high, 2),
        "CustomerSegment": ref_cat
    })

    current_df = pd.DataFrame({
        "StableMetric": np.round(cur_stable, 2),
        "ModerateMetric": np.round(cur_mod, 2),
        "HighShiftMetric": np.round(cur_high, 2),
        "CustomerSegment": cur_cat
    })

    return reference_df, current_df


def save_default_samples(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    quality_df = generate_sample_quality_dataset()
    quality_path = os.path.join(output_dir, "sample_quality_data.csv")
    quality_df.to_csv(quality_path, index=False)

    ref_df, cur_df = generate_sample_drift_datasets()
    ref_path = os.path.join(output_dir, "sample_reference_drift.csv")
    cur_path = os.path.join(output_dir, "sample_current_drift.csv")
    ref_df.to_csv(ref_path, index=False)
    cur_df.to_csv(cur_path, index=False)

    return quality_path, ref_path, cur_path
