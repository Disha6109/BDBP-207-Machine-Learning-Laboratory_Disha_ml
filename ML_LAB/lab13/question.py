import numpy as np

from sklearn.datasets import load_diabetes, load_iris
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# =========================================
# 1. LOAD DATASETS
# =========================================

# Regression → Diabetes
data_reg = load_diabetes()
Xr, yr = data_reg.data, data_reg.target

# Classification → Iris
data_clf = load_iris()
Xc, yc = data_clf.data, data_clf.target

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2)

# =========================================
# 2. BAGGING (SKLEARN)
# =========================================

# --- Regression
bag_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=10
)
bag_reg.fit(Xr_train, yr_train)

pred = bag_reg.predict(Xr_test)
print("Bagging Regressor MSE:", mean_squared_error(yr_test, pred))


# --- Classification
bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10
)
bag_clf.fit(Xc_train, yc_train)

pred = bag_clf.predict(Xc_test)
print("Bagging Classifier Accuracy:", accuracy_score(yc_test, pred))


# =========================================
# 3. BAGGING FROM SCRATCH (REGRESSION)
# =========================================

class SimpleBagging:
    def __init__(self, n_estimators=5):
        self.n = n_estimators
        self.models = []

    def fit(self, X, y):
        n_samples = len(X)

        for _ in range(self.n):
            # Bootstrap sampling
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idx]
            y_sample = y[idx]

            model = DecisionTreeRegressor()
            model.fit(X_sample, y_sample)

            self.models.append(model)

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        return np.mean(preds, axis=0)   # average

# Train scratch bagging
bag = SimpleBagging(n_estimators=5)
bag.fit(Xr_train, yr_train)

pred = bag.predict(Xr_test)
print("Scratch Bagging MSE:", mean_squared_error(yr_test, pred))


# =========================================
# 4. RANDOM FOREST (SKLEARN)
# =========================================

# --- Regression
rf_reg = RandomForestRegressor(n_estimators=10)
rf_reg.fit(Xr_train, yr_train)

pred = rf_reg.predict(Xr_test)
print("Random Forest Regressor MSE:", mean_squared_error(yr_test, pred))


# --- Classification
rf_clf = RandomForestClassifier(n_estimators=10)
rf_clf.fit(Xc_train, yc_train)

pred = rf_clf.predict(Xc_test)
print("Random Forest Classifier Accuracy:", accuracy_score(yc_test, pred))