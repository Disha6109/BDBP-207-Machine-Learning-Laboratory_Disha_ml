import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

# ==============================
# 1. LOAD DATASET
# ==============================

data = pd.read_csv("sonar.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert labels: Metal (M)=1, Rock (R)=0
y = np.where(y == 'M', 1, 0)

# ==============================
# 2. NORMALIZATION (SCRATCH)
# ==============================

def normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)

# ==============================
# 3. STANDARDIZATION (SCRATCH)
# ==============================

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)

# ==============================
# 4. K-FOLD SPLIT (SCRATCH)
# ==============================

def k_fold_split(X, y, k=10):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    fold_size = len(X) // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size

        test_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        folds.append((train_idx, test_idx))

    return folds

# ==============================
# 5. K-FOLD CV (SCRATCH)
# ==============================

def k_fold_cv(X, y, k=10):
    folds = k_fold_split(X, y, k)
    scores = []

    for train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        scores.append(acc)

    return np.mean(scores)

# ==============================
# 6. RESULTS
# ==============================

print("\n===== RESULTS =====")

# --- Without preprocessing (SCRATCH CV)
acc_raw_scratch = k_fold_cv(X, y, k=10)
print("Scratch CV (No preprocessing):", acc_raw_scratch)

# --- With normalization (SCRATCH CV)
X_norm = normalize(X)
acc_norm_scratch = k_fold_cv(X_norm, y, k=10)
print("Scratch CV (Normalization):", acc_norm_scratch)

# --- With standardization (SCRATCH CV)
X_std = standardize(X)
acc_std_scratch = k_fold_cv(X_std, y, k=10)
print("Scratch CV (Standardization):", acc_std_scratch)

# ==============================
# 7. SKLEARN METHODS
# ==============================

model = LogisticRegression(max_iter=1000)
kf = KFold(n_splits=10, shuffle=True)

# --- Without preprocessing
scores_raw = cross_val_score(model, X, y, cv=kf)
print("\nsklearn CV (No preprocessing):", scores_raw.mean())

# --- With normalization (sklearn)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

scores_scaled = cross_val_score(model, X_scaled, y, cv=kf)
print("sklearn CV (Normalization):", scores_scaled.mean())

# --- With standardization (sklearn)
std_scaler = StandardScaler()
X_std2 = std_scaler.fit_transform(X)

scores_std = cross_val_score(model, X_std2, y, cv=kf)
print("sklearn CV (Standardization):", scores_std.mean())