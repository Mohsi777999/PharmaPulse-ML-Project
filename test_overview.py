"""
Unit and Integration Tests for Operations Overview Engine
Tests dynamic filtering, KPI mathematics, time range scoping, category slicing,
and cross-module intelligence alerts.
"""

import sys
import os
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.routers.overview import get_overview_data
from modules.intelligence.insights import get_pharmacy_intelligence_summary


class TestOverviewIntelligence(unittest.TestCase):

    def test_01_default_overview_schema_and_kpis(self):
        """Test default invocation returns complete schema and valid KPI ranges."""
        res = get_overview_data()
        self.assertIn("kpis", res)
        self.assertIn("available_categories", res)
        self.assertIn("monthly_velocity", res)
        self.assertIn("categories", res)
        self.assertIn("critical_expiry_batches", res)
        self.assertIn("urgent_stockouts", res)
        self.assertIn("top_growing", res)
        self.assertIn("top_declining", res)

        kpis = res["kpis"]
        self.assertEqual(kpis["active_medicines"], 150)
        self.assertGreater(kpis["gross_revenue_m"], 200.0)
        self.assertGreater(kpis["gross_profit_m"], 50.0)
        self.assertGreater(kpis["gross_margin_pct"], 20.0)
        self.assertLess(kpis["gross_margin_pct"], 50.0)
        self.assertGreater(kpis["fulfillment_rate_pct"], 90.0)
        self.assertGreater(kpis["stockout_days"], 1000)
        self.assertGreater(kpis["stockout_pct"], 0.0)
        self.assertEqual(len(res["available_categories"]), 7)  # ALL + 6 categories

    def test_02_dynamic_time_range_scoping(self):
        """Test that time range filters (12M, 6M, 30D) scale metrics correctly."""
        res_all = get_overview_data(time_range="ALL")
        res_12m = get_overview_data(time_range="12M")
        res_6m = get_overview_data(time_range="6M")
        res_30d = get_overview_data(time_range="30D")

        # Revenue should scale down as time window narrows
        rev_all = res_all["kpis"]["gross_revenue_m"]
        rev_12m = res_12m["kpis"]["gross_revenue_m"]
        rev_6m = res_6m["kpis"]["gross_revenue_m"]
        rev_30d = res_30d["kpis"]["gross_revenue_m"]

        self.assertGreater(rev_all, rev_12m)
        self.assertGreater(rev_12m, rev_6m)
        self.assertGreater(rev_6m, rev_30d)

        # Monthly velocity counts
        self.assertEqual(len(res_all["monthly_velocity"]), 36)
        self.assertEqual(len(res_12m["monthly_velocity"]), 12)
        self.assertEqual(len(res_6m["monthly_velocity"]), 6)
        # 30D should provide daily resolution
        self.assertGreaterEqual(len(res_30d["monthly_velocity"]), 28)

    def test_03_dynamic_category_filtering(self):
        """Test filtering by category reduces active formulary and isolates category data."""
        res_cat1 = get_overview_data(category_id="CAT001")
        kpis = res_cat1["kpis"]

        self.assertEqual(kpis["selected_category_id"], "CAT001")
        self.assertEqual(kpis["selected_category_name"], "Pain relief and fever")
        self.assertEqual(kpis["active_medicines"], 30)

        # Category breakdown in single-category view shows top medicines in that category
        breakdown = res_cat1["categories"]
        self.assertGreater(len(breakdown), 0)
        for item in breakdown:
            self.assertEqual(item["sku_count"], 1)
            self.assertGreater(item["units_sold"], 0)

    def test_04_expiry_batches_deduplication(self):
        """Test critical expiry batches are unique by batch_id with valid exposure."""
        res = get_overview_data()
        batches = res["critical_expiry_batches"]
        self.assertGreater(len(batches), 0)

        batch_ids = [b["batch_id"] for b in batches]
        self.assertEqual(len(batch_ids), len(set(batch_ids)), "Batch IDs in critical list must be unique (no duplicates).")

        for b in batches:
            self.assertLessEqual(b["days_to_expiry"], 30)
            self.assertGreater(b["financial_exposure"], 0)
            self.assertIn(b["target_expiry_risk_class"], ["High", "Critical"])

    def test_05_urgent_stockouts_validity(self):
        """Test stockout replenishment alerts contain populated, realistic operational fields."""
        res = get_overview_data()
        stockouts = res["urgent_stockouts"]
        self.assertGreater(len(stockouts), 0, "Urgent stockout alerts must not be empty.")

        for s in stockouts:
            self.assertIn("purchase_order_id", s)
            self.assertIn("medicine_name", s)
            self.assertIn("days_of_supply", s)
            self.assertIn("expected_lead_time_days", s)
            self.assertIn("supplier_id", s)
            self.assertIn("urgency_status", s)
            self.assertGreater(s["expected_lead_time_days"], 0)

    def test_06_cross_filter_combination(self):
        """Test combining both category and time_range filters simultaneously."""
        res = get_overview_data(time_range="12M", category_id="CAT005")
        kpis = res["kpis"]

        self.assertEqual(kpis["active_medicines"], 25)
        self.assertEqual(kpis["selected_category_id"], "CAT005")
        self.assertEqual(kpis["time_range"], "12M")
        self.assertGreater(kpis["gross_revenue_m"], 5.0)
        self.assertGreater(kpis["units_sold"], 100000)
        self.assertEqual(len(res["monthly_velocity"]), 12)


if __name__ == "__main__":
    unittest.main()
