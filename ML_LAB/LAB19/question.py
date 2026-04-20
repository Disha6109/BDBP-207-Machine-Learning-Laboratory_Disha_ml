import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_csv("heart.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values   # target (0/1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================================
# 2. TRAIN MODEL
# =========================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# probabilities
probs = model.predict_proba(X_test)[:, 1]

# =========================================
# 3. METRICS FROM SCRATCH
# =========================================

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def metrics(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-9)
    sensitivity = TP / (TP + FN + 1e-9)   # recall
    specificity = TN / (TN + FP + 1e-9)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-9)

    return accuracy, precision, sensitivity, specificity, f1

# =========================================
# 4. VARY THRESHOLDS
# =========================================

thresholds = [0.3, 0.5, 0.7]

for t in thresholds:
    y_pred = (probs >= t).astype(int)

    acc, prec, sens, spec, f1 = metrics(y_test, y_pred)

    print(f"\nThreshold = {t}")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Sensitivity:", sens)
    print("Specificity:", spec)
    print("F1 Score:", f1)

# =========================================
# 5. ROC CURVE + AUC (FROM SCRATCH)
# =========================================

tpr_list = []
fpr_list = []

thresh_range = np.linspace(0, 1, 50)

for t in thresh_range:
    y_pred = (probs >= t).astype(int)

    TP, TN, FP, FN = confusion_matrix(y_test, y_pred)

    TPR = TP / (TP + FN + 1e-9)   # sensitivity
    FPR = FP / (FP + TN + 1e-9)

    tpr_list.append(TPR)
    fpr_list.append(FPR)

# Sort for proper plotting
fpr_list, tpr_list = zip(*sorted(zip(fpr_list, tpr_list)))

# AUC (trapezoidal rule)
auc = np.trapz(tpr_list, fpr_list)

# Plot ROC
plt.plot(fpr_list, tpr_list, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("\nAUC:", auc)