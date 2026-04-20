import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# ===================================
# 1. L1 and L2 Norm (Simple)
# ===================================

w = np.array([1, -2, 3])

l1 = np.sum(np.abs(w))
l2 = np.sqrt(np.sum(w**2))

print("L1 Norm:", l1)
print("L2 Norm:", l2)


# ===================================
# 2. Simple Encoding (from scratch)
# ===================================

data = ['cat', 'dog', 'cat']

# Ordinal Encoding
unique = list(set(data))
ord_map = {val:i for i, val in enumerate(unique)}
ord_encoded = [ord_map[i] for i in data]

print("\nOrdinal:", ord_encoded)

# One-hot Encoding
onehot = []
for i in data:
    row = [0]*len(unique)
    row[ord_map[i]] = 1
    onehot.append(row)

print("One-hot:", onehot)


# ===================================
# 3. Wisconsin Dataset (Ridge & Lasso)
# ===================================

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Ridge
ridge = RidgeClassifier()
ridge.fit(X_train, y_train)
pred = ridge.predict(X_test)
print("\nRidge Accuracy:", accuracy_score(y_test, pred))

# Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
pred = lasso.predict(X_test)
pred = [1 if i > 0.5 else 0 for i in pred]
print("Lasso Accuracy:", accuracy_score(y_test, pred))


# ===================================
# 4. Breast Cancer CSV (Encoding + LR)
# ===================================

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
df = pd.read_csv(url, header=None)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Remove missing values
X = X.replace('?', np.nan)
X = X.dropna()
y = y[X.index]

# Label Encoding (target)
le = LabelEncoder()
y = le.fit_transform(y)

# Ordinal Encoding
oe = OrdinalEncoder()
X_ord = oe.fit_transform(X)

# One-Hot Encoding
ohe = OneHotEncoder(sparse=False)
X_ohe = ohe.fit_transform(X)

# Logistic Regression (Ordinal)
X_train, X_test, y_train, y_test = train_test_split(X_ord, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nLR (Ordinal):", accuracy_score(y_test, pred))

# Logistic Regression (One-hot)
X_train, X_test, y_train, y_test = train_test_split(X_ohe, y, test_size=0.2)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("LR (One-hot):", accuracy_score(y_test, pred))