import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# =========================================
# LAB 17 - DATA
# =========================================

X = np.array([
    [1,13],[1,18],[2,9],[3,6],[6,3],[9,2],[13,1],[18,1],
    [3,15],[6,6],[6,11],[9,5],[10,10],[11,5],[12,6],[16,3]
])

labels = ['Blue']*8 + ['Red']*8
y = np.array([0 if i=='Blue' else 1 for i in labels])

# =========================================
# 1. PLOT ORIGINAL DATA
# =========================================

plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Original Data (2D)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# =========================================
# 2. TRANSFORM FUNCTION (3D)
# =========================================

def transform(X):
    x1 = X[:,0]
    x2 = X[:,1]
    return np.column_stack((x1**2, np.sqrt(2)*x1*x2, x2**2))

X_trans = transform(X)

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_trans[:,0], X_trans[:,1], X_trans[:,2], c=y)
ax.set_title("Transformed Data (3D)")
plt.show()

# =========================================
# 3. DOT PRODUCT CHECK
# =========================================

x1 = np.array([3,6])
x2 = np.array([10,10])

def transform_single(x):
    return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])

dot_high = np.dot(transform_single(x1), transform_single(x2))
print("\nDot product in higher dimension:", dot_high)

# Polynomial Kernel
def poly_kernel(a,b):
    return (a[0]**2)*(b[0]**2) + 2*a[0]*b[0]*a[1]*b[1] + (a[1]**2)*(b[1]**2)

dot_kernel = poly_kernel(x1, x2)
print("Polynomial kernel result:", dot_kernel)


# =========================================
# LAB 18 - RBF vs POLY KERNEL
# =========================================

X2 = np.array([
    [6,5],[6,9],[8,6],[8,8],[8,10],[9,2],[9,5],
    [10,10],[10,13],[11,5],[11,8],[12,6],[12,11],
    [13,4],[14,8]
])

labels2 = ['Blue','Blue','Red','Red','Red','Blue','Red',
           'Red','Blue','Red','Red','Red','Blue','Blue','Blue']

y2 = np.array([0 if i=='Blue' else 1 for i in labels2])

# Polynomial Kernel SVM
model_poly = SVC(kernel='poly', degree=2)
model_poly.fit(X2, y2)

pred_poly = model_poly.predict(X2)
print("\nPoly Kernel Accuracy:", accuracy_score(y2, pred_poly))

# RBF Kernel SVM
model_rbf = SVC(kernel='rbf')
model_rbf.fit(X2, y2)

pred_rbf = model_rbf.predict(X2)
print("RBF Kernel Accuracy:", accuracy_score(y2, pred_rbf))


# =========================================
# IRIS DATASET (CLASS 1 vs 2)
# =========================================

iris = load_iris()
X = iris.data[:, :2]   # first 2 features
y = iris.target

# Take only class 1 and 2
mask = (y == 1) | (y == 2)
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

pred = svm.predict(X_test)
print("\nIris SVM Accuracy:", accuracy_score(y_test, pred))