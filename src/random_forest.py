from numbers import Integral
from typing import Callable

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _matrix(X, n_features=None, feature_names=None):
    if feature_names is not None and hasattr(X, "columns"):
        if tuple(X.columns) != feature_names:
            raise ValueError("Feature names and order differ from training")
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("X must be a two-dimensional matrix with features")
    if n_features is not None and values.shape[1] != n_features:
        raise ValueError("Wrong number of features")
    if not np.isfinite(values).all():
        raise ValueError("X must contain finite values")
    return values


def _target(y, n_samples):
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != n_samples or n_samples == 0:
        raise ValueError("y must be one-dimensional and match nonempty X")
    classes, encoded = np.unique(values, return_inverse=True)
    if not 1 <= len(classes) <= 2:
        raise ValueError("Only binary or single-class targets are supported")
    return values, classes, encoded


class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=None, min_samples_leaf=1,
                 max_features="sqrt", random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        _positive_integer(self.min_samples_leaf, "min_samples_leaf")
        if self.max_depth is not None:
            _positive_integer(self.max_depth, "max_depth")
        self.feature_names_in_ = tuple(X.columns) if hasattr(X, "columns") else None
        values = _matrix(X)
        _, self.classes_, encoded = _target(y, len(values))
        self.n_features_in_ = values.shape[1]
        if self.max_features == "sqrt":
            self.max_features_ = max(1, int(np.sqrt(self.n_features_in_)))
        elif self.max_features is None:
            self.max_features_ = self.n_features_in_
        else:
            _positive_integer(self.max_features, "max_features")
            if self.max_features > self.n_features_in_:
                raise ValueError("max_features exceeds the number of features")
            self.max_features_ = int(self.max_features)

        weights = np.ones(len(values)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(values),) or not np.isfinite(weights).all():
            raise ValueError("Invalid sample weights")
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Sample weights must be nonnegative with positive sum")

        self._X = values
        self._y = encoded
        self._weights = weights
        self._rng = np.random.RandomState(self.random_state)
        self.children_left_ = []
        self.children_right_ = []
        self.feature_ = []
        self.threshold_ = []
        self.positive_probability_ = []
        self.n_node_samples_ = []
        self.node_depth_ = []
        self._grow(np.flatnonzero(weights > 0), depth=0)
        for name in ("children_left_", "children_right_", "feature_", "n_node_samples_", "node_depth_"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.intp))
        self.threshold_ = np.asarray(self.threshold_, dtype=float)
        self.positive_probability_ = np.asarray(self.positive_probability_, dtype=float)
        self.depth_ = int(self.node_depth_.max())
        del self._X, self._y, self._weights, self._rng
        return self

    def _best_split(self, samples, features):
        values = self._X[np.ix_(samples, features)]
        order = np.argsort(values, axis=0, kind="stable")
        sorted_values = np.take_along_axis(values, order, axis=0)
        weights = self._weights[samples]
        positive_weights = weights * self._y[samples]
        left_weight = np.cumsum(weights[order], axis=0)[:-1]
        left_positive = np.cumsum(positive_weights[order], axis=0)[:-1]
        right_weight = weights.sum() - left_weight
        right_positive = positive_weights.sum() - left_positive

        loss = (
            left_positive * (left_weight - left_positive) / left_weight
            + right_positive * (right_weight - right_positive) / right_weight
        )
        left_size = np.arange(1, len(samples))[:, None]
        valid = (
            (sorted_values[:-1] < sorted_values[1:])
            & (left_size >= self.min_samples_leaf)
            & (len(samples) - left_size >= self.min_samples_leaf)
        )
        loss[~valid] = np.inf
        best_by_feature = loss.min(axis=0)
        column = int(np.argmin(best_by_feature))
        if not np.isfinite(best_by_feature[column]):
            return None
        row = int(np.argmin(loss[:, column]))
        low, high = float(sorted_values[row, column]), float(sorted_values[row + 1, column])
        return int(features[column]), low / 2 + high / 2

    def _grow(self, samples, depth):
        weights = self._weights[samples]
        total = weights.sum()
        positive = np.dot(weights, self._y[samples])
        node = len(self.feature_)
        self.children_left_.append(-1)
        self.children_right_.append(-1)
        self.feature_.append(-1)
        self.threshold_.append(np.nan)
        self.positive_probability_.append(positive / total)
        self.n_node_samples_.append(len(samples))
        self.node_depth_.append(depth)
        if (positive == 0 or positive == total
                or len(samples) < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node

        features = self._rng.permutation(self.n_features_in_)
        split = self._best_split(samples, features[:self.max_features_])
        if split is None:
            for feature in features[self.max_features_:]:
                split = self._best_split(samples, np.array([feature]))
                if split is not None:
                    break
        if split is None:
            return node
        feature, threshold = split
        goes_left = self._X[samples, feature] <= threshold
        self.feature_[node] = feature
        self.threshold_[node] = threshold
        self.children_left_[node] = self._grow(samples[goes_left], depth + 1)
        self.children_right_[node] = self._grow(samples[~goes_left], depth + 1)
        return node

    def predict_proba(self, X):
        if not hasattr(self, "classes_"):
            raise ValueError("Fit the tree before prediction")
        values = _matrix(X, self.n_features_in_, self.feature_names_in_)
        nodes = np.zeros(len(values), dtype=np.intp)
        while True:
            active = np.flatnonzero(self.feature_[nodes] >= 0)
            if len(active) == 0:
                break
            current = nodes[active]
            left = values[active, self.feature_[current]] <= self.threshold_[current]
            nodes[active] = np.where(left, self.children_left_[current], self.children_right_[current])
        if len(self.classes_) == 1:
            return np.ones((len(values), 1))
        positive = self.positive_probability_[nodes]
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        probability = self.predict_proba(X)
        return self.classes_[probability.argmax(axis=1)]


class RandomForestClassifierScratch:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1,
                 max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y, on_tree: Callable | None = None):
        _positive_integer(self.n_estimators, "n_estimators")
        _positive_integer(self.min_samples_leaf, "min_samples_leaf")
        if self.max_depth is not None:
            _positive_integer(self.max_depth, "max_depth")
        self.feature_names_in_ = tuple(X.columns) if hasattr(X, "columns") else None
        values = _matrix(X)
        target, self.classes_, _ = _target(y, len(values))
        self.n_features_in_ = values.shape[1]
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self.estimator_seeds_ = []
        for tree_number in range(self.n_estimators):
            seed = int(rng.randint(np.iinfo(np.int32).max))
            sampled = np.random.RandomState(seed).randint(0, len(values), size=len(values))
            weights = np.bincount(sampled, minlength=len(values))
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            ).fit(values, target, sample_weight=weights)
            self.estimators_.append(tree)
            self.estimator_seeds_.append(seed)
            if on_tree is not None:
                on_tree(tree_number + 1, tree)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "estimators_") or not self.estimators_:
            raise ValueError("Fit the forest before prediction")
        values = _matrix(X, self.n_features_in_, self.feature_names_in_)
        result = np.zeros((len(values), len(self.classes_)))
        for tree in self.estimators_:
            result += tree.predict_proba(values)
        return result / len(self.estimators_)

    def predict(self, X):
        probability = self.predict_proba(X)
        return self.classes_[probability.argmax(axis=1)]
