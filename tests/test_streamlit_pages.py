import sys
import unittest
from streamlit.testing.v1 import AppTest

class TestStreamlitPages(unittest.TestCase):

    def test_all_pages_render(self):
        pages = [
            "📊 Overview Dashboard",
            "🔍 Data Quality Analysis",
            "✍️ Human Approval & Override",
            "🧠 Anomaly Detection",
            "📈 Data Drift Analysis",
            "⚖️ Quality Comparison",
            "📑 Combined Reports"
        ]

        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        self.assertFalse(at.exception, "Default overview page produced an exception")

        radio = at.sidebar.radio[0]

        for p in pages:
            radio.set_value(p)
            at.run(timeout=30)
            self.assertFalse(at.exception, f"Page '{p}' produced an exception: {at.exception}")

if __name__ == "__main__":
    unittest.main()
