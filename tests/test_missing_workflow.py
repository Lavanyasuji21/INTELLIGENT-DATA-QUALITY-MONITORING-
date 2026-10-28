import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.core_engine import DataQualityEngine
from backend.sample_data import generate_sample_quality_dataset

class TestMissingValueWorkflow(unittest.TestCase):

    def setUp(self):
        self.engine = DataQualityEngine("test_missing_unit")
        self.df = generate_sample_quality_dataset()
        self.engine.analyze_dataset(self.df)

    def test_01_missing_records_structure(self):
        records = self.engine.APP.get("missing_records", [])
        self.assertGreater(len(records), 0)

        # Check fields of each record
        for r in records:
            self.assertIn("cell_id", r)
            self.assertIn("row_idx", r)
            self.assertIn("row_num", r)
            self.assertEqual(r["row_num"], r["row_idx"] + 1)
            self.assertIn("column", r)
            self.assertEqual(r["current_value"], "Missing (NaN)")
            self.assertIn("suggested_value", r)
            self.assertIn("status", r)
            self.assertEqual(r["status"], "Pending")

        # Verify DataFrame format
        df_records = self.engine.get_missing_records_df()
        self.assertIn("Row Number", df_records.columns)
        self.assertIn("Column Name", df_records.columns)
        self.assertIn("Current Value", df_records.columns)
        self.assertIn("Suggested Replacement", df_records.columns)
        self.assertIn("Approval Status", df_records.columns)
        self.assertEqual(len(df_records), len(records))

    def test_02_missing_summary(self):
        summary = self.engine.get_missing_summary()
        self.assertGreater(summary["total_missing_cells"], 0)
        self.assertGreater(summary["rows_with_missing"], 0)
        self.assertGreater(summary["cols_with_missing"], 0)
        self.assertFalse(summary["col_breakdown"].empty)
        self.assertIn("Column Name", summary["col_breakdown"].columns)
        self.assertIn("Missing Cells", summary["col_breakdown"].columns)

    def test_03_individual_approve_valid_numeric(self):
        # Find first Age record (numeric)
        age_records = [r for r in self.engine.APP["missing_records"] if r["column"] == "Age"]
        self.assertTrue(len(age_records) > 0)
        rec = age_records[0]

        # Initially, cell in approved dataset is missing (NaN)
        row_idx = rec["row_idx"]
        self.assertTrue(pd.isna(self.engine.APP["approved"].iloc[row_idx]["Age"]))

        # Approve with a custom value 25
        success, msg = self.engine.update_missing_record(rec["cell_id"], "25", "Approved")
        self.assertTrue(success)
        self.assertIn("Approved replacement for Row", msg)

        # Approved dataset cell must now equal 25
        self.assertEqual(self.engine.APP["approved"].iloc[row_idx]["Age"], 25)

        # Audit record must be updated
        self.assertEqual(rec["status"], "Approved")
        self.assertEqual(rec["replacement_value"], "25")
        self.assertEqual(rec["approved_value"], "25")

    def test_04_individual_approve_invalid_numeric(self):
        # Find first Age record (numeric)
        age_records = [r for r in self.engine.APP["missing_records"] if r["column"] == "Age"]
        rec = age_records[0]

        # Attempt to approve with non-numeric text "TwentyFive"
        success, msg = self.engine.update_missing_record(rec["cell_id"], "TwentyFive", "Approved")
        self.assertFalse(success)
        self.assertIn("Invalid numeric value", msg)
        # Record status should still be Pending and cell should still be NaN
        self.assertEqual(rec["status"], "Pending")
        self.assertTrue(pd.isna(self.engine.APP["approved"].iloc[rec["row_idx"]]["Age"]))

    def test_05_individual_reject(self):
        age_records = [r for r in self.engine.APP["missing_records"] if r["column"] == "Age"]
        rec = age_records[0]

        success, msg = self.engine.update_missing_record(rec["cell_id"], "", "Rejected")
        self.assertTrue(success)
        self.assertEqual(rec["status"], "Rejected")
        # Cell in approved dataset remains missing (NaN)
        self.assertTrue(pd.isna(self.engine.APP["approved"].iloc[rec["row_idx"]]["Age"]))

    def test_06_batch_approve_and_reject(self):
        # Approve all missing records
        app_cnt, fail_cnt, msg = self.engine.approve_all_missing_records()
        self.assertGreater(app_cnt, 0)
        self.assertEqual(fail_cnt, 0)

        # Check no missing cells left in approved dataset for those columns
        for r in self.engine.APP["missing_records"]:
            self.assertEqual(r["status"], "Approved")
            row_idx = r["row_idx"]
            col = r["column"]
            self.assertFalse(pd.isna(self.engine.APP["approved"].iloc[row_idx][col]))

        # Now reject all missing records
        rej_cnt, msg2 = self.engine.reject_all_missing_records()
        self.assertGreater(rej_cnt, 0)

        for r in self.engine.APP["missing_records"]:
            self.assertEqual(r["status"], "Rejected")
            row_idx = r["row_idx"]
            col = r["column"]
            self.assertTrue(pd.isna(self.engine.APP["approved"].iloc[row_idx][col]))

    def test_07_sync_with_general_actions(self):
        # Test accept_suggestions bulk sync
        self.engine.approve_suggestions()
        for r in self.engine.APP["missing_records"]:
            self.assertEqual(r["status"], "Approved")

        # Test reject_suggestions bulk sync
        self.engine.reject_suggestions()
        for r in self.engine.APP["missing_records"]:
            self.assertEqual(r["status"], "Rejected")

        # Test manual override on a missing cell
        rec = self.engine.APP["missing_records"][0]
        self.engine.apply_human_override(rec["row_idx"], rec["column"], "999")
        self.assertEqual(rec["status"], "Approved")
        self.assertEqual(rec["approved_value"], "999")

if __name__ == "__main__":
    unittest.main()
