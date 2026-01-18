
X = [
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
]


X_T = [
    [1, 0, 2, 1, 0],
    [0, 1, 1, 1, 2],
    [2, 1, 0, 1, 1]
]


n = len(X)


cov = [[0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]

for i in range(3):
    for j in range(3):
        for k in range(n):
            cov[i][j] += X_T[i][k] * X[k][j]
        cov[i][j] = cov[i][j] / n


print("Covariance Matrix:")
for row in cov:
    print(row)

#To check using numpy
import numpy as np

X_npy = np.array(X)


n = len(X)


covariance_np = np.dot(X_npy.T, X_npy) / n

print("Covariance matrix using NumPy:")
print(covariance_np)
