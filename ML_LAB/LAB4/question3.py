import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# --------------------------------------------------

# Step 1: Load Dataset

# --------------------------------------------------

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")


# Features and target

X = data.drop(columns=["disease_score", "disease_score_fluct"]).values

y = data["disease_score_fluct"].values.reshape(-1, 1)


m = len(y)


# Add bias column for scratch methods

X_bias = np.hstack((np.ones((m, 1)), X))


# --------------------------------------------------

# Step 2: Gradient Descent Functions

# --------------------------------------------------

def hypothesis(X, theta):

    return np.dot(X, theta)


def compute_cost(X, y, theta):

    m = len(y)

    return (1/(2*m)) * np.sum((hypothesis(X, theta) - y) ** 2)


def compute_gradient(X, y, theta):

    m = len(y)

    return (1/m) * np.dot(X.T, (hypothesis(X, theta) - y))


def gradient_descent(X, y, theta, lr, iterations):

    cost_history = []

    for _ in range(iterations):

        theta = theta - lr * compute_gradient(X, y, theta)

        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


# Run Gradient Descent

theta_init = np.zeros((X_bias.shape[1], 1))

theta_gd, cost_history = gradient_descent(X_bias, y, theta_init, lr=0.0001, iterations=2000)


# Predictions using Gradient Descent

pred_gd = hypothesis(X_bias, theta_gd)


# --------------------------------------------------

# Step 3: Normal Equation

# --------------------------------------------------

theta_ne = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

pred_ne = hypothesis(X_bias, theta_ne)


# --------------------------------------------------

# Step 4: Scikit-Learn Model

# --------------------------------------------------

model = LinearRegression()

model.fit(X, y)

pred_sk = model.predict(X)


# --------------------------------------------------

# Step 5: Compare Results

# --------------------------------------------------

print("R2 Score (Gradient Descent):", r2_score(y, pred_gd))

print("R2 Score (Normal Equation):", r2_score(y, pred_ne))

print("R2 Score (Scikit-Learn):", r2_score(y, pred_sk))


# --------------------------------------------------

# Step 6: Plot Cost Reduction (GD)

# --------------------------------------------------

plt.figure()

plt.plot(cost_history)

plt.xlabel("Iterations")

plt.ylabel("Cost")

plt.title("Gradient Descent Cost Reduction")

plt.show()




