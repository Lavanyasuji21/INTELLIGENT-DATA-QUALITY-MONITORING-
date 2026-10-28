import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.core_engine import DataQualityEngine, get_session
from backend.sample_data import generate_sample_quality_dataset, generate_sample_drift_datasets

class TestDataQualityEngine(unittest.TestCase):

    def setUp(self):
        self.session_id = "test_unit_session"
        self.engine = get_session(self.session_id)
        self.test_data_dir = os.path.join(BASE_DIR, "data", "test_tmp")
        os.makedirs(self.test_data_dir, exist_ok=True)

        self.quality_df = generate_sample_quality_dataset()
        self.quality_csv_path = os.path.join(self.test_data_dir, "test_quality.csv")
        self.quality_df.to_csv(self.quality_csv_path, index=False)

        self.ref_df, self.cur_df = generate_sample_drift_datasets()
        self.ref_csv_path = os.path.join(self.test_data_dir, "test_ref.csv")
        self.cur_csv_path = os.path.join(self.test_data_dir, "test_cur.csv")
        self.ref_df.to_csv(self.ref_csv_path, index=False)
        self.cur_df.to_csv(self.cur_csv_path, index=False)

    def test_01_quality_metrics_calculation(self):
        metrics = self.engine.calculate_quality_metrics(self.quality_df)
        self.assertIn("Rows", metrics)
        self.assertIn("Columns", metrics)
        self.assertIn("Missing Cells", metrics)
        self.assertIn("Duplicate Rows", metrics)
        self.assertIn("Entirely Empty Columns", metrics)
        self.assertIn("Quality Score (%)", metrics)
        self.assertGreater(metrics["Missing Cells"], 0)
        self.assertEqual(metrics["Duplicate Rows"], 2)
        self.assertGreaterEqual(metrics["Quality Score (%)"], 0)
        self.assertLessEqual(metrics["Quality Score (%)"], 100)

    def test_02_analyze_dataset(self):
        summary, issues, orig, prop, appr = self.engine.analyze_dataset(self.quality_csv_path)
        self.assertIn("Dataset loaded successfully", summary)
        self.assertFalse(issues.empty)
        self.assertFalse(orig.empty)
        self.assertFalse(prop.empty)
        self.assertFalse(appr.empty)

        # Check issues contain detected problems
        problems = issues["Problem"].unique()
        self.assertIn("Missing Values", problems)
        self.assertIn("Duplicate Rows", problems)

        # State check
        self.assertIsNotNone(self.engine.APP["original"])
        self.assertIsNotNone(self.engine.APP["proposed"])
        # Initially, approved must match original until accepted
        self.assertEqual(len(self.engine.APP["approved"]), len(self.engine.APP["original"]))

    def test_03_human_approval_workflow(self):
        self.engine.analyze_dataset(self.quality_csv_path)

        # 1. Accept suggestions
        msg, appr_preview = self.engine.approve_suggestions()
        self.assertIn("accepted", msg)
        # Missing values in Age should now be imputed in approved
        self.assertEqual(int(self.engine.APP["approved"]["Age"].isna().sum()), 0)

        # 2. Reject suggestions
        msg, appr_preview = self.engine.reject_suggestions()
        self.assertIn("rejected", msg)
        self.assertGreater(int(self.engine.APP["approved"]["Age"].isna().sum()), 0)

        # 3. Accept suggestions again for further testing
        self.engine.approve_suggestions()

        # 4. Remove duplicates
        before_count = len(self.engine.APP["approved"])
        msg, appr_preview = self.engine.remove_duplicates()
        self.assertIn("Removed", msg)
        self.assertEqual(len(self.engine.APP["approved"]), before_count - 2)

        # 5. Apply human override
        msg, appr_preview = self.engine.apply_human_override(0, "Age", 99.0)
        self.assertIn("Override applied", msg)
        self.assertEqual(float(self.engine.APP["approved"].iloc[0]["Age"]), 99.0)

    def test_04_quality_comparison(self):
        self.engine.analyze_dataset(self.quality_csv_path)
        self.engine.approve_suggestions()
        self.engine.remove_duplicates()

        summary, comp_df, plot_path = self.engine.compare_quality_before_after()
        self.assertIn("Before-and-after comparison completed", summary)
        self.assertFalse(comp_df.empty)
        self.assertTrue(os.path.exists(plot_path))

    def test_05_anomaly_detection(self):
        summary, result_df, plot_path = self.engine.run_anomaly_detection(
            self.quality_csv_path, percentile=95.0
        )
        self.assertIn("Anomaly detection completed", summary)
        self.assertIn("Autoencoder_Anomaly", result_df.columns)
        self.assertIn("IsolationForest_Anomaly", result_df.columns)
        self.assertIn("Top_Contributing_Features", result_df.columns)
        self.assertTrue(os.path.exists(plot_path))

    def test_06_data_drift_analysis(self):
        summary, drift_df, plot_path = self.engine.run_drift_analysis(
            self.ref_csv_path, self.cur_csv_path
        )
        self.assertIn("Data drift analysis completed", summary)
        self.assertFalse(drift_df.empty)
        self.assertIn("PSI", drift_df.columns)
        self.assertIn("Drift Level", drift_df.columns)
        self.assertIn("Statistical Result", drift_df.columns)
        self.assertTrue(os.path.exists(plot_path))

    def test_07_excel_reports(self):
        self.engine.analyze_dataset(self.quality_csv_path)
        self.engine.approve_suggestions()
        self.engine.run_anomaly_detection(self.quality_csv_path, percentile=90.0)
        self.engine.run_drift_analysis(self.ref_csv_path, self.cur_csv_path)
        self.engine.compare_quality_before_after()

        # Download reports
        q_path = self.engine.download_quality_report()
        self.assertTrue(os.path.exists(q_path))

        appr_path = self.engine.download_approved_dataset()
        self.assertTrue(os.path.exists(appr_path))

        anom_path = self.engine.download_anomaly_report()
        self.assertTrue(os.path.exists(anom_path))

        drift_path = self.engine.download_drift_report()
        self.assertTrue(os.path.exists(drift_path))

        comb_path = self.engine.download_combined_report()
        self.assertTrue(os.path.exists(comb_path))


if __name__ == "__main__":
    unittest.main()
