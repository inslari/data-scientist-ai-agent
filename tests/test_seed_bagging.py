"""Offline tests for the multi-seed bagging layer in ml_trainer (no LLM/DB).

Covers the picklable seed-bag wrappers, the random_state fan-out, the gating
helper (`_maybe_seed_bag`) and seed-bag feature-importance extraction.
"""

import pickle
import unittest

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.config import settings
from src.services.ml_trainer import (
    SeedBaggingClassifier,
    SeedBaggingRegressor,
    _maybe_seed_bag,
    _seed_list,
    _set_all_random_states,
    _extract_feature_importance,
    _extra_classification_models,
    _get_classification_models,
    _get_classification_models_balanced,
    _SEED_BAGGABLE,
    _SLOW_MODELS,
    _SLOW_MODEL_MAX_ROUNDS,
)


def _clf_data():
    return make_classification(
        n_samples=200, n_features=8, n_informative=5, random_state=0
    )


def _reg_data():
    return make_regression(n_samples=200, n_features=8, n_informative=5, random_state=0)


class SeedListTest(unittest.TestCase):
    def test_consecutive_from_random_seed(self):
        seeds = _seed_list(3)
        self.assertEqual(len(seeds), 3)
        self.assertEqual(seeds[0], int(settings.RANDOM_SEED))
        self.assertEqual(seeds, [seeds[0], seeds[0] + 1, seeds[0] + 2])

    def test_floor_of_one(self):
        self.assertEqual(len(_seed_list(0)), 1)


class SetRandomStateTest(unittest.TestCase):
    def test_sets_nested_pipeline_random_state(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=1)),
        ])
        _set_all_random_states(pipe, 99)
        self.assertEqual(pipe.get_params()["clf__random_state"], 99)

    def test_no_random_state_is_safe(self):
        knn = KNeighborsClassifier()           # no random_state param
        # must not raise and must leave the estimator usable
        _set_all_random_states(knn, 7)
        X, y = _clf_data()
        knn.fit(X, y)
        self.assertEqual(len(knn.predict(X)), len(y))


class SeedBaggingClassifierTest(unittest.TestCase):
    def test_fits_one_member_per_seed(self):
        X, y = _clf_data()
        bag = SeedBaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=8, random_state=42),
            seeds=[42, 43, 44],
        )
        bag.fit(X, y)
        self.assertEqual(len(bag.fitted_), 3)
        self.assertEqual(list(bag.classes_), [0, 1])

    def test_predict_proba_is_member_average(self):
        X, y = _clf_data()
        seeds = [42, 43, 44]
        bag = SeedBaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=8, random_state=42),
            seeds=seeds,
        ).fit(X, y)
        manual = np.mean([m.predict_proba(X) for m in bag.fitted_], axis=0)
        np.testing.assert_allclose(bag.predict_proba(X), manual, rtol=1e-6)
        # proba rows sum to 1 and predict matches argmax
        np.testing.assert_allclose(bag.predict_proba(X).sum(axis=1), 1.0, rtol=1e-6)
        np.testing.assert_array_equal(
            bag.predict(X), bag.classes_[np.argmax(bag.predict_proba(X), axis=1)]
        )

    def test_distinct_seeds_produce_distinct_members(self):
        X, y = _clf_data()
        bag = SeedBaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=8, random_state=42),
            seeds=[1, 2],
        ).fit(X, y)
        p0 = bag.fitted_[0].predict_proba(X)
        p1 = bag.fitted_[1].predict_proba(X)
        self.assertFalse(np.allclose(p0, p1))   # different seeds → different fits

    def test_picklable(self):
        X, y = _clf_data()
        bag = SeedBaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=8, random_state=42),
            seeds=[42, 43],
        ).fit(X, y)
        restored = pickle.loads(pickle.dumps(bag))
        np.testing.assert_allclose(restored.predict_proba(X), bag.predict_proba(X))

    def test_sample_weight_passthrough(self):
        X, y = _clf_data()
        sw = np.ones(len(y))
        bag = SeedBaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=8, random_state=42),
            seeds=[42, 43],
        )
        bag.fit(X, y, sample_weight=sw)        # must not raise
        self.assertEqual(len(bag.fitted_), 2)


class SeedBaggingRegressorTest(unittest.TestCase):
    def test_predict_is_member_average(self):
        X, y = _reg_data()
        bag = SeedBaggingRegressor(
            base_estimator=RandomForestRegressor(n_estimators=8, random_state=42),
            seeds=[42, 43, 44],
        ).fit(X, y)
        manual = np.mean([m.predict(X) for m in bag.fitted_], axis=0)
        np.testing.assert_allclose(bag.predict(X), manual, rtol=1e-6)

    def test_picklable(self):
        X, y = _reg_data()
        bag = SeedBaggingRegressor(
            base_estimator=RandomForestRegressor(n_estimators=8, random_state=42),
            seeds=[42, 43],
        ).fit(X, y)
        restored = pickle.loads(pickle.dumps(bag))
        np.testing.assert_allclose(restored.predict(X), bag.predict(X))


class MaybeSeedBagGatingTest(unittest.TestCase):
    def setUp(self):
        # snapshot + restore so we don't leak settings into other tests
        self._enabled = settings.SEED_BAGGING_ENABLED
        self._rounds = settings.SEED_BAGGING_ROUNDS

    def tearDown(self):
        settings.SEED_BAGGING_ENABLED = self._enabled
        settings.SEED_BAGGING_ROUNDS = self._rounds

    def test_wraps_seed_sensitive_model_when_enabled(self):
        settings.SEED_BAGGING_ENABLED = True
        settings.SEED_BAGGING_ROUNDS = 3
        base = RandomForestClassifier(n_estimators=5)
        model, rounds = _maybe_seed_bag(base, "random_forest", "classification")
        self.assertIsInstance(model, SeedBaggingClassifier)
        self.assertEqual(rounds, 3)

    def test_regression_returns_regressor_wrapper(self):
        settings.SEED_BAGGING_ENABLED = True
        settings.SEED_BAGGING_ROUNDS = 2
        model, rounds = _maybe_seed_bag(
            RandomForestRegressor(n_estimators=5), "random_forest", "regression"
        )
        self.assertIsInstance(model, SeedBaggingRegressor)
        self.assertEqual(rounds, 2)

    def test_skips_non_seed_sensitive_model(self):
        settings.SEED_BAGGING_ENABLED = True
        settings.SEED_BAGGING_ROUNDS = 3
        base = KNeighborsClassifier()
        model, rounds = _maybe_seed_bag(base, "knn", "classification")
        self.assertIs(model, base)             # unchanged
        self.assertEqual(rounds, 0)
        self.assertNotIn("knn", _SEED_BAGGABLE)

    def test_skips_when_rounds_below_two(self):
        settings.SEED_BAGGING_ENABLED = True
        settings.SEED_BAGGING_ROUNDS = 1
        base = RandomForestClassifier(n_estimators=5)
        model, rounds = _maybe_seed_bag(base, "random_forest", "classification")
        self.assertIs(model, base)
        self.assertEqual(rounds, 0)

    def test_skips_when_disabled(self):
        settings.SEED_BAGGING_ENABLED = False
        settings.SEED_BAGGING_ROUNDS = 3
        base = RandomForestClassifier(n_estimators=5)
        model, rounds = _maybe_seed_bag(base, "random_forest", "classification")
        self.assertIs(model, base)
        self.assertEqual(rounds, 0)

    def test_slow_model_rounds_capped(self):
        settings.SEED_BAGGING_ENABLED = True
        settings.SEED_BAGGING_ROUNDS = 3
        # catboost is slow + baggable → rounds capped to _SLOW_MODEL_MAX_ROUNDS
        _, rounds = _maybe_seed_bag(RandomForestClassifier(n_estimators=5), "catboost", "classification")
        self.assertEqual(rounds, _SLOW_MODEL_MAX_ROUNDS)
        self.assertIn("catboost", _SLOW_MODELS)
        # fast model keeps the full round count
        _, fast_rounds = _maybe_seed_bag(RandomForestClassifier(n_estimators=5), "random_forest", "classification")
        self.assertEqual(fast_rounds, 3)


class SeedBagImportanceTest(unittest.TestCase):
    def test_importance_averaged_over_members(self):
        X, y = _clf_data()
        names = [f"f{i}" for i in range(X.shape[1])]
        bag = SeedBaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            seeds=[42, 43, 44],
        ).fit(X, y)
        imp = _extract_feature_importance(bag, names)
        self.assertEqual(set(imp), set(names))
        # importances are non-negative and sum ~1 (RF feature_importances_ average)
        self.assertTrue(all(v >= 0 for v in imp.values()))
        self.assertAlmostEqual(sum(imp.values()), 1.0, places=4)


class ExtraModelZooTest(unittest.TestCase):
    _EXTRA = {"decision_tree", "naive_bayes", "knn", "svm", "mlp"}

    def setUp(self):
        self._extra_enabled = settings.EXTRA_MODELS_ENABLED

    def tearDown(self):
        settings.EXTRA_MODELS_ENABLED = self._extra_enabled

    def test_helper_returns_all_baselines(self):
        names = set(_extra_classification_models(balanced=False))
        self.assertEqual(names, self._EXTRA)

    def test_merged_into_both_catalogs_when_enabled(self):
        settings.EXTRA_MODELS_ENABLED = True
        plain = set(_get_classification_models())
        balanced = set(_get_classification_models_balanced({0: 1.0, 1: 1.0}))
        self.assertTrue(self._EXTRA.issubset(plain))
        self.assertTrue(self._EXTRA.issubset(balanced))

    def test_absent_when_disabled(self):
        settings.EXTRA_MODELS_ENABLED = False
        plain = set(_get_classification_models())
        self.assertFalse(self._EXTRA & plain)

    def test_svm_exposes_predict_proba_after_fit(self):
        # LinearSVC has no predict_proba; the calibrated wrapper must provide it
        # (stacking / OOF blend depend on it).
        X, y = _clf_data()
        svm = _extra_classification_models(balanced=False)["svm"]
        svm.fit(X, y)
        proba = svm.predict_proba(X)
        self.assertEqual(proba.shape, (len(y), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    def test_balanced_sets_class_weight_only_where_supported(self):
        bal = _extra_classification_models(balanced=True)
        self.assertEqual(bal["decision_tree"].get_params()["class_weight"], "balanced")
        self.assertEqual(bal["svm"].get_params()["clf__estimator__class_weight"], "balanced")

    def test_baseline_models_are_seed_bag_aware(self):
        # decision_tree/mlp are seed-sensitive → baggable; knn/naive_bayes are not.
        self.assertIn("decision_tree", _SEED_BAGGABLE)
        self.assertIn("mlp", _SEED_BAGGABLE)
        self.assertNotIn("knn", _SEED_BAGGABLE)
        self.assertNotIn("naive_bayes", _SEED_BAGGABLE)


if __name__ == "__main__":
    unittest.main()
