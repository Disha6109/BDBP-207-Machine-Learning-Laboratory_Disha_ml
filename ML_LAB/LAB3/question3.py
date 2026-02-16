

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


# -----------------------------

# Step 1: start with reading the csv file

# -----------------------------

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")


# -----------------------------

# Step 2: Define X and y

# -----------------------------

X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
y = data["disease_score_fluct"].values.reshape(-1, 1)

m = len(y)


X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# -----------------------------

# Step 3: Defining the Hypothesis Function

# -----------------------------

def hypothesis(X, theta):

    return np.dot(X, theta)


# -----------------------------

# Step 4: Computing Cost Function

# -----------------------------

def compute_cost(X, y, theta):

    m = len(y)

    predictions = hypothesis(X, theta)

    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

    return cost


# -----------------------------

# Step 5: Computing Derivative

# -----------------------------

def compute_gradient(X, y, theta):

    m = len(y)

    predictions = hypothesis(X, theta)

    gradient = (1 / m) * np.dot(X.T, (predictions - y))

    return gradient


# -----------------------------

# Step 6: Computing Gradient Descent

# -----------------------------

def gradient_descent(X, y, theta, learning_rate, iterations):

    cost_history = []


    for i in range(iterations):

        gradient = compute_gradient(X, y, theta)

        theta = theta - learning_rate * gradient

        cost_history.append(compute_cost(X, y, theta))


    return theta, cost_history


theta = np.zeros((X.shape[1], 1))

learning_rate = 0.01

iterations = 1000

theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)


# -----------------------------

# Step 7: Calculating the results

# -----------------------------

print("Final Parameters (Theta):\n", theta_final)

print("Final Cost:", cost_history[-1])


# -----------------------------

# Step 8: Plotting the Results

# -----------------------------

plt.plot(cost_history)

plt.xlabel("Iterations")

plt.ylabel("Cost")

plt.title("Cost Reduction Over Iterations")

plt.show()
