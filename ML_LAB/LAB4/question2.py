import numpy as np

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error


# Load data

data = fetch_california_housing()

X, y = data.data, data.target


# Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

def gradient_descent(X, y, lr=0.01, itr=2000):
    m = len(y)                     # number of samples
    X_b = np.c_[np.ones(m), X]     # add bias column
    n = X_b.shape[1]               # number of features (+1 for bias)

    theta = np.zeros(n)            # initialize parameters

    for _ in range(itr):
        y_pred = X_b.dot(theta)
        error = y_pred - y
        gradients = (1/m) * X_b.T.dot(error)
        theta -= lr * gradients

    return theta


# Train using Gradient Descent

theta = gradient_descent(X_train, y_train, lr=0.01, itr=2000)


# Predict

X_test_b = np.c_[np.ones(len(X_test)), X_test]

y_pred_gd = X_test_b.dot(theta)


# Train using Sklearn

model = LinearRegression()

model.fit(X_train, y_train)

y_pred_sk = model.predict(X_test)


print("California Housing MSE (GD):", mean_squared_error(y_test, y_pred_gd))

print("California Housing MSE (Sklearn):", mean_squared_error(y_test, y_pred_sk))
