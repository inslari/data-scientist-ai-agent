"""Tests for deterministic shared-key relationship augmentation (GROUP 11 layout).

Reconstructs the real 5-CSV competition data group and proves the churn label
table gets linked to the feature tables on cust_id, while the test/submission
artifacts are excluded — so the executor's join pulls churn in.
"""

import types
import unittest

from src.intelligence_engine import (
    _augment_shared_key_relationships,
    _is_artifact_table,
    _looks_like_join_key,
)
from src.services.data_executor import _resolve_all_related_tables, _build_merge_order


def _col(table, name, semantic_type="metric"):
    return types.SimpleNamespace(table_name=table, column_name=name, semantic_type=semantic_type)


def _fake_model():
    """Mirror GROUP 11: 2 feature tables + label + test split + submission."""
    cols = [
        # customer_history (feature, fact — 5.3M rows)
        _col("customer_history", "cust_id", "identifier"),
        _col("customer_history", "date", "timestamp"),
        _col("customer_history", "mobile_eft_all_cnt"),
        # customers (feature, demographic)
        _col("customers", "cust_id", "identifier"),
        _col("customers", "age"),
        # referance_data (TRAIN LABEL)
        _col("referance_data", "cust_id", "identifier"),
        _col("referance_data", "ref_date", "timestamp"),
        _col("referance_data", "churn", "target"),
        # referance_data_test (TEST split — must be excluded)
        _col("referance_data_test", "cust_id", "identifier"),
        _col("referance_data_test", "ref_date", "timestamp"),
        # sample_submission (submission template — must be excluded)
        _col("sample_submission", "cust_id", "identifier"),
        _col("sample_submission", "churn", "target"),
    ]
    return types.SimpleNamespace(columns=cols, relationships=[])


class ArtifactDetectionTest(unittest.TestCase):
    def test_artifact_tables(self):
        self.assertTrue(_is_artifact_table("referance_data_test"))
        self.assertTrue(_is_artifact_table("sample_submission"))
        self.assertTrue(_is_artifact_table("holdout_set"))

    def test_non_artifact_tables(self):
        self.assertFalse(_is_artifact_table("customer_history"))
        self.assertFalse(_is_artifact_table("customers"))
        self.assertFalse(_is_artifact_table("referance_data"))
        self.assertFalse(_is_artifact_table("latest_orders"))  # 'test' substring, not a word


class JoinKeyDetectionTest(unittest.TestCase):
    def test_identifier_and_id_suffix(self):
        self.assertTrue(_looks_like_join_key("cust_id", "identifier"))
        self.assertTrue(_looks_like_join_key("account_id", "metric"))
        self.assertTrue(_looks_like_join_key("id", "unknown"))

    def test_non_keys(self):
        self.assertFalse(_looks_like_join_key("churn", "target"))
        self.assertFalse(_looks_like_join_key("amount", "metric"))
        self.assertFalse(_looks_like_join_key("valid", "metric"))  # not an *_id


class AugmentTest(unittest.TestCase):
    def setUp(self):
        self.model = _augment_shared_key_relationships(_fake_model())
        self.rels = [
            (r.source_table, r.target_table) for r in self.model.relationships
        ]
        self.linked = {t for pair in self.rels for t in pair}

    def test_label_table_linked_to_features(self):
        # referance_data must be connected to the feature side via cust_id
        self.assertIn("referance_data", self.linked)
        self.assertIn("customer_history", self.linked)
        self.assertIn("customers", self.linked)

    def test_artifacts_excluded(self):
        self.assertNotIn("referance_data_test", self.linked)
        self.assertNotIn("sample_submission", self.linked)

    def test_all_relationships_on_cust_id(self):
        for r in self.model.relationships:
            self.assertEqual(r.source_column, "cust_id")
            self.assertEqual(r.target_column, "cust_id")

    def test_idempotent(self):
        n1 = len(self.model.relationships)
        _augment_shared_key_relationships(self.model)
        self.assertEqual(len(self.model.relationships), n1)


class ExecutorJoinReachabilityTest(unittest.TestCase):
    """Prove the executor would now pull the churn label into the join."""

    def setUp(self):
        model = _augment_shared_key_relationships(_fake_model())
        self.rels = [
            {"source_table": r.source_table, "source_column": r.source_column,
             "target_table": r.target_table, "target_column": r.target_column}
            for r in model.relationships
        ]
        self.all_known = ["customer_history", "customers", "referance_data",
                          "referance_data_test", "sample_submission"]

    def test_join_reaches_label_not_artifacts(self):
        # Planner anchors on the fact table; graph walk should reach label + features
        reached = _resolve_all_related_tables(["customer_history"], self.rels, self.all_known)
        self.assertIn("referance_data", reached)
        self.assertIn("customers", reached)
        self.assertNotIn("referance_data_test", reached)
        self.assertNotIn("sample_submission", reached)

    def test_merge_order_includes_label(self):
        reached = _resolve_all_related_tables(["customer_history"], self.rels, self.all_known)
        ops = _build_merge_order(reached, self.rels, anchor="customer_history")
        joined_tables = {op["right"] for op in ops}
        self.assertIn("referance_data", joined_tables)  # churn label gets joined


if __name__ == "__main__":
    unittest.main()
