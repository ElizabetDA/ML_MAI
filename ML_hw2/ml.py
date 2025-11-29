import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    x = np.asarray(feature_vector, dtype=float)
    y = np.asarray(target_vector, dtype=int)

    n = len(x)
    if n <= 1:
        return np.array([]), np.array([]), None, None

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    unequal = x_sorted[1:] != x_sorted[:-1]
    if not np.any(unequal):
        return np.array([]), np.array([]), None, None

    thresholds = (x_sorted[1:][unequal] + x_sorted[:-1][unequal]) / 2.0

    prefix_pos = np.cumsum(y_sorted)
    total_pos = prefix_pos[-1]
    total_cnt = n
    idx = np.nonzero(unequal)[0]

    left_cnt = idx + 1
    right_cnt = total_cnt - left_cnt

    left_pos = prefix_pos[idx]
    right_pos = total_pos - left_pos

    p_left = left_pos / left_cnt
    p_right = right_pos / right_cnt

    Hl = 1 - p_left**2 - (1 - p_left)**2
    Hr = 1 - p_right**2 - (1 - p_right)**2

    ginis = -(left_cnt/total_cnt * Hl + right_cnt/total_cnt * Hr)

    best = np.argmax(ginis)
    return thresholds, ginis, thresholds[best], ginis[best]


class DecisionTree:

    def __init__(self, feature_types,
                 max_depth=None,
                 min_samples_split=None,
                 min_samples_leaf=None):

        # sklearn clone требует:
        # 1) не изменять feature_types
        # 2) хранить параметр в self.<имя>, совпадающее с названием
        self.feature_types = feature_types  

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self._tree = {}

    # clone вызывает get_params() → конструктор
    def get_params(self, deep=True):
        return {
            "feature_types": self.feature_types,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def _make_leaf(self, y, node):
        node["type"] = "leaf"
        node["class"] = int(Counter(y).most_common(1)[0][0])

    def _should_stop(self, X, y, depth):
        if np.all(y == y[0]):
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if self.min_samples_split is not None and len(y) < self.min_samples_split:
            return True
        return False

    def _fit_node(self, X, y, node, depth):
        n, m = X.shape

        if self._should_stop(X, y, depth):
            self._make_leaf(y, node)
            return

        best_feature = None
        best_threshold = None
        best_split = None
        best_gini = None
        best_cats = None

        for j in range(m):
            ftype = self.feature_types[j]
            col = X[:, j]

            # categorical
            if ftype == "categorical":
                counts = Counter(col)
                pos_counts = Counter(col[y == 1])
                ratios = {c: pos_counts.get(c, 0) / counts[c] for c in counts}
                sorted_cats = sorted(ratios.items(), key=lambda x: x[1])
                mapping = {cat: i for i, (cat, _) in enumerate(sorted_cats)}
                feat = np.array([mapping[c] for c in col], dtype=float)
            else:
                feat = col.astype(float)
                mapping = None

            if np.all(feat == feat[0]):
                continue

            _, _, thr, gini = find_best_split(feat, y)
            if thr is None:
                continue

            split = feat <= thr
            lc = split.sum()
            rc = n - lc

            if self.min_samples_leaf is not None:
                if lc < self.min_samples_leaf or rc < self.min_samples_leaf:
                    continue

            if best_gini is None or gini > best_gini:
                best_gini = gini
                best_feature = j
                best_threshold = thr
                best_split = split
                if ftype == "categorical":
                    best_cats = [cat for cat, k in mapping.items() if k <= thr]
                else:
                    best_cats = None

        if best_feature is None:
            self._make_leaf(y, node)
            return

        node["type"] = "node"
        node["feature"] = best_feature

        if self.feature_types[best_feature] == "real":
            node["threshold"] = float(best_threshold)
        else:
            node["categories"] = best_cats

        node["left"] = {}
        node["right"] = {}

        self._fit_node(X[best_split], y[best_split], node["left"], depth + 1)
        self._fit_node(X[~best_split], y[~best_split], node["right"], depth + 1)

    def _predict_one(self, x, node):
        if node["type"] == "leaf":
            return node["class"]

        j = node["feature"]
        if self.feature_types[j] == "real":
            if x[j] <= node["threshold"]:
                return self._predict_one(x, node["left"])
            else:
                return self._predict_one(x, node["right"])
        else:
            if x[j] in node["categories"]:
                return self._predict_one(x, node["left"])
            else:
                return self._predict_one(x, node["right"])

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(np.asarray(X), np.asarray(y), self._tree, 0)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(row, self._tree) for row in X], dtype=int)
