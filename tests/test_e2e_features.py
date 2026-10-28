import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.core_engine import DataQualityEngine, get_session
from backend.sample_data import generate_sample_quality_dataset, generate_sample_drift_datasets

class TestComprehensiveE2E(unittest.TestCase):

    def setUp(self):
        self.session_a = "user_alpha"
        self.session_b = "user_beta"
        self.engine_a = DataQualityEngine(self.session_a)
        self.engine_b = DataQualityEngine(self.session_b)

        self.tmp_dir = os.path.join(BASE_DIR, "data", "test_tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.quality_df = generate_sample_quality_dataset()
        self.quality_csv = os.path.join(self.tmp_dir, "test_quality.csv")
        self.quality_df.to_csv(self.quality_csv, index=False)

        self.ref_df, self.cur_df = generate_sample_drift_datasets()
        self.ref_csv = os.path.join(self.tmp_dir, "test_ref.csv")
        self.cur_csv = os.path.join(self.tmp_dir, "test_cur.csv")
        self.ref_df.to_csv(self.ref_csv, index=False)
        self.cur_df.to_csv(self.cur_csv, index=False)

    def test_01_upload_and_analyze(self):
        summary, issues, orig, prop, appr = self.engine_a.analyze_dataset(self.quality_csv)
        self.assertIn("Dataset loaded successfully", summary)
        self.assertGreater(len(issues), 0)
        self.assertEqual(len(orig), 15)
        self.assertEqual(len(prop), 15)
        self.assertEqual(len(appr), 15)
        self.assertEqual(len(self.engine_a.APP["original"]), 62)

    def test_02_detect_missing_values_and_issues(self):
        self.engine_a.analyze_dataset(self.quality_csv)
        issues = self.engine_a.APP["issues"]
        problems = set(issues["Problem"].unique())
        
        self.assertIn("Missing Values", problems)
        self.assertIn("Duplicate Rows", problems)
        self.assertIn("Negative Values", problems)
        self.assertIn("Potential Outliers (IQR)", problems)
        self.assertIn("Constant Column", problems)
        self.assertIn("Numbers Stored as Text", problems)
        self.assertIn("Leading/Trailing Spaces", problems)

    def test_03_accept_and_reject_suggestions(self):
        self.engine_a.analyze_dataset(self.quality_csv)
        
        # Test Accept
        msg_acc, _ = self.engine_a.approve_suggestions()
        self.assertIn("accepted", msg_acc)
        self.assertEqual(self.engine_a.APP["approved"]["Age"].isna().sum(), 0)
        
        # Test Reject
        msg_rej, _ = self.engine_a.reject_suggestions()
        self.assertIn("rejected", msg_rej)
        self.assertGreater(self.engine_a.APP["approved"]["Age"].isna().sum(), 0)

    def test_04_duplicate_handling(self):
        self.engine_a.analyze_dataset(self.quality_csv)
        self.assertEqual(self.engine_a.APP["approved"].duplicated().sum(), 2)
        
        # Keep duplicates
        msg_keep, _ = self.engine_a.keep_duplicates()
        self.assertIn("Kept", msg_keep)
        self.assertEqual(self.engine_a.APP["approved"].duplicated().sum(), 2)
        
        # Remove duplicates
        msg_rem, _ = self.engine_a.remove_duplicates()
        self.assertIn("Removed 2 duplicate", msg_rem)
        self.assertEqual(self.engine_a.APP["approved"].duplicated().sum(), 0)

    def test_05_manual_human_override(self):
        self.engine_a.analyze_dataset(self.quality_csv)
        self.engine_a.approve_suggestions()
        
        # Apply override on row 0, column Age
        msg, _ = self.engine_a.apply_human_override(0, "Age", 99.0)
        self.assertIn("Override applied", msg)
        self.assertEqual(self.engine_a.APP["approved"].loc[0, "Age"], 99.0)
        
        # Invalid row index
        msg_err, _ = self.engine_a.apply_human_override(9999, "Age", 50)
        self.assertIn("must be between 0", msg_err)
        
        # Invalid column
        msg_col_err, _ = self.engine_a.apply_human_override(0, "NonExistentColumn", 50)
        self.assertIn("not found", msg_col_err)

    def test_06_autoencoder_threshold_adjustment(self):
        # Run anomaly detection with default threshold (95)
        sum_95, res_95, plot_95 = self.engine_a.run_anomaly_detection(self.quality_csv, percentile=95.0)
        self.assertIsNotNone(res_95)
        self.assertTrue(os.path.exists(plot_95))
        ae_count_95 = int(res_95["Autoencoder_Anomaly"].sum())
        
        # Run with lower threshold (85) -> should flag more or equal anomalies
        sum_85, res_85, plot_85 = self.engine_a.run_anomaly_detection(self.quality_csv, percentile=85.0)
        ae_count_85 = int(res_85["Autoencoder_Anomaly"].sum())
        self.assertGreaterEqual(ae_count_85, ae_count_95)
        
        # Check explanation features
        self.assertIn("Top_Contributing_Features", res_95.columns)
        self.assertTrue(res_95["Top_Contributing_Features"].str.contains("error=").any())

    def test_07_data_drift_analysis(self):
        summary, drift_df, plot_path = self.engine_a.run_drift_analysis(self.ref_csv, self.cur_csv)
        self.assertIn("Data drift analysis completed", summary)
        self.assertTrue(os.path.exists(plot_path))
        
        # Check columns and drift levels
        self.assertIn("PSI", drift_df.columns)
        self.assertIn("Drift Level", drift_df.columns)
        self.assertIn("Statistical Test", drift_df.columns)
        self.assertIn("P-Value", drift_df.columns)
        
        drift_levels = set(drift_df["Drift Level"].unique())
        self.assertTrue("Low" in drift_levels or "Moderate" in drift_levels or "High" in drift_levels)

    def test_08_before_after_quality_comparison(self):
        self.engine_a.analyze_dataset(self.quality_csv)
        self.engine_a.approve_suggestions()
        self.engine_a.remove_duplicates()
        
        summary, comp_df, plot_path = self.engine_a.compare_quality_before_after()
        self.assertIn("Before-and-after comparison completed", summary)
        self.assertTrue(os.path.exists(plot_path))
        
        metrics = set(comp_df["Metric"].unique())
        self.assertIn("Quality Score (%)", metrics)
        self.assertIn("Missing Cells", metrics)
        self.assertIn("Duplicate Rows", metrics)

    def test_09_excel_reports_generation(self):
        self.engine_a.analyze_dataset(self.quality_csv)
        self.engine_a.approve_suggestions()
        self.engine_a.run_anomaly_detection(self.quality_csv, percentile=95.0)
        self.engine_a.run_drift_analysis(self.ref_csv, self.cur_csv)
        self.engine_a.compare_quality_before_after()
        
        # 1. Quality report
        q_path = self.engine_a.download_quality_report()
        self.assertTrue(os.path.exists(q_path))
        
        # 2. Approved dataset
        a_path = self.engine_a.download_approved_dataset()
        self.assertTrue(os.path.exists(a_path))
        
        # 3. Anomaly report
        an_path = self.engine_a.download_anomaly_report()
        self.assertTrue(os.path.exists(an_path))
        
        # 4. Drift report
        dr_path = self.engine_a.download_drift_report()
        self.assertTrue(os.path.exists(dr_path))
        
        # 5. Combined report
        cb_path = self.engine_a.download_combined_report()
        self.assertTrue(os.path.exists(cb_path))

    def test_10_empty_and_invalid_file_handling(self):
        # Empty file
        empty_csv = os.path.join(self.tmp_dir, "empty.csv")
        with open(empty_csv, "w") as f:
            f.write("")
        summary, _, _, _, _ = self.engine_a.analyze_dataset(empty_csv)
        self.assertIn("Upload error", summary)
        
        # Unsupported format
        unsupported = os.path.join(self.tmp_dir, "test.txt")
        with open(unsupported, "w") as f:
            f.write("Some text")
        summary2, _, _, _, _ = self.engine_a.analyze_dataset(unsupported)
        self.assertIn("Supported formats are CSV, XLS and XLSX", summary2)

    def test_11_session_isolation(self):
        # User Alpha loads a dataset
        self.engine_a.analyze_dataset(self.quality_csv)
        self.assertIsNotNone(self.engine_a.APP["original"])
        
        # User Beta should have completely empty state
        self.assertIsNone(self.engine_b.APP["original"])
        self.assertIsNone(self.engine_b.APP["approved"])
        
        # User Beta performs drift analysis
        self.engine_b.run_drift_analysis(self.ref_csv, self.cur_csv)
        self.assertIsNotNone(self.engine_b.APP["drift_report"])
        
        # User Alpha's drift report is still None
        self.assertIsNone(self.engine_a.APP["drift_report"])

if __name__ == "__main__":
    unittest.main()
