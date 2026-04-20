import numpy as np

from sklearn.datasets import load_iris, load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# =========================================
# 1. LOAD DATA (IRIS)
# =========================================

data = load_iris()
X = data.data
y = data.target

# Binary conversion (NO astype)
y = np.where(y == 0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================================
# LAB 14 - ADABOOST (SKLEARN)
# =========================================

ada = AdaBoostClassifier(n_estimators=10)
ada.fit(X_train, y_train)

pred = ada.predict(X_test)
print("AdaBoost (sklearn) Accuracy:", accuracy_score(y_test, pred))


# =========================================
# LAB 14 - ADABOOST (SCRATCH)
# =========================================

class SimpleAdaBoost:
    def __init__(self, n_estimators=5):
        self.n = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n = len(y)
        w = np.ones(n) / n   # initial weights

        for _ in range(self.n):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)

            pred = stump.predict(X)

            error = np.sum(w * (pred != y))

            # avoid division error
            if error == 0:
                alpha = 1
            else:
                alpha = 0.5 * np.log((1 - error) / (error + 1e-9))

            # update weights
            w = w * np.exp(-alpha * (2*y-1) * (2*pred-1))
            w = w / np.sum(w)

            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        final = np.zeros(len(X))

        for alpha, model in zip(self.alphas, self.models):
            pred = model.predict(X)
            final += alpha * (2*pred - 1)

        return (final > 0).astype(int)


ada_scratch = SimpleAdaBoost(n_estimators=5)
ada_scratch.fit(X_train, y_train)

pred = ada_scratch.predict(X_test)
print("AdaBoost (scratch) Accuracy:", accuracy_score(y_test, pred))


# =========================================
# LAB 15 - GRADIENT BOOSTING
# =========================================

# Classification
gb_clf = GradientBoostingClassifier(n_estimators=50)
gb_clf.fit(X_train, y_train)

pred = gb_clf.predict(X_test)
print("\nGradient Boost (Classifier):", accuracy_score(y_test, pred))


# Regression (Diabetes dataset)
data_reg = load_diabetes()
Xr, yr = data_reg.data, data_reg.target

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2)

gb_reg = GradientBoostingRegressor(n_estimators=50)
gb_reg.fit(Xr_train, yr_train)

pred = gb_reg.predict(Xr_test)
print("Gradient Boost (Regressor MSE):", mean_squared_error(yr_test, pred))


# =========================================
# LAB 16 - AGGREGATION (FROM SCRATCH)
# =========================================

tree1 = np.array([100, 120, 130])
tree2 = np.array([110, 115, 125])
tree3 = np.array([105, 118, 128])

final_pred = (tree1 + tree2 + tree3) / 3
print("\nAggregated Prediction:", final_pred)


# =========================================
# LAB 16 - XGBOOST
# =========================================

try:
    from xgboost import XGBClassifier, XGBRegressor

    # Classification
    xgb_clf = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)

    pred = xgb_clf.predict(X_test)
    print("XGBoost Classifier:", accuracy_score(y_test, pred))

    # Regression
    xgb_reg = XGBRegressor(n_estimators=50)
    xgb_reg.fit(Xr_train, yr_train)

    pred = xgb_reg.predict(Xr_test)
    print("XGBoost Regressor MSE:", mean_squared_error(yr_test, pred))

except:
    print("\nInstall xgboost if needed: pip install xgboost")