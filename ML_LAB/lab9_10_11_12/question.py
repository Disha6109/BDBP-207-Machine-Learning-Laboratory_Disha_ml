import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# =========================================
# LAB 9 - PARTITION DATASET
# =========================================

# Simulated data
data = pd.DataFrame({
    'BP': [70, 75, 80, 85, 90],
    'target': [100, 110, 120, 130, 140]
})

def partition(df, t):
    left = df[df['BP'] <= t]
    right = df[df['BP'] > t]
    print(f"\nThreshold = {t}")
    print("Left:\n", left)
    print("Right:\n", right)

partition(data, 80)
partition(data, 78)
partition(data, 82)


# =========================================
# LAB 9 - DECISION TREE REGRESSOR (sklearn)
# =========================================

X = data[['BP']]
y = data['target']

model = DecisionTreeRegressor()
model.fit(X, y)

pred = model.predict(X)
print("\nRegression MSE:", mean_squared_error(y, pred))


# =========================================
# LAB 9 - DECISION TREE CLASSIFIER (SONAR)
# =========================================

# Load sonar dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
df = pd.read_csv(url, header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert labels
y = np.where(y == 'M', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("\nSonar Accuracy:", accuracy_score(y_test, pred))


# =========================================
# LAB 10 - ENTROPY
# =========================================

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return -np.sum(prob * np.log2(prob + 1e-9))

# Example
y_sample = np.array([0, 0, 1, 1, 1])
print("\nEntropy:", entropy(y_sample))


# =========================================
# LAB 10 - INFORMATION GAIN
# =========================================

def information_gain(parent, left, right):
    H_parent = entropy(parent)

    w_left = len(left) / len(parent)
    w_right = len(right) / len(parent)

    IG = H_parent - (w_left * entropy(left) + w_right * entropy(right))
    return IG

# Example
left = np.array([0, 0])
right = np.array([1, 1, 1])
print("Information Gain:", information_gain(y_sample, left, right))


# =========================================
# LAB 11 - DECISION TREE CLASSIFIER (SCRATCH)
# =========================================

class SimpleTree:
    def fit(self, X, y):
        # choose best feature (simple: first feature)
        self.feature = 0
        self.threshold = np.mean(X[:, 0])

        self.left_class = np.bincount(y[X[:, 0] <= self.threshold]).argmax()
        self.right_class = np.bincount(y[X[:, 0] > self.threshold]).argmax()

    def predict(self, X):
        preds = []
        for x in X:
            if x[self.feature] <= self.threshold:
                preds.append(self.left_class)
            else:
                preds.append(self.right_class)
        return np.array(preds)

# Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Binary classification (simplify)
y = (y == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree = SimpleTree()
tree.fit(X_train, y_train)

pred = tree.predict(X_test)
print("\nScratch Classifier Accuracy:", accuracy_score(y_test, pred))


# =========================================
# LAB 12 - DECISION TREE REGRESSOR (SCRATCH)
# =========================================

class SimpleTreeRegressor:
    def fit(self, X, y):
        self.feature = 0
        self.threshold = np.mean(X[:, 0])

        self.left_value = np.mean(y[X[:, 0] <= self.threshold])
        self.right_value = np.mean(y[X[:, 0] > self.threshold])

    def predict(self, X):
        preds = []
        for x in X:
            if x[self.feature] <= self.threshold:
                preds.append(self.left_value)
            else:
                preds.append(self.right_value)
        return np.array(preds)

# Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree_reg = SimpleTreeRegressor()
tree_reg.fit(X_train, y_train)

pred = tree_reg.predict(X_test)
print("Scratch Regressor MSE:", mean_squared_error(y_test, pred))