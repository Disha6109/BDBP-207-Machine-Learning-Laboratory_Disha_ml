

import numpy as np


from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



def gradient_descent(X, y, lr=0.01, itr =1000):

    m, n = X.shape

    X = np.c_[np.ones(m), X]

    theta = np.zeros(n + 1)


    for  i in range(itr):

        predictions = X.dot(theta)

        errors = predictions - y

        gradients = (1/m) * X.T.dot(errors)

        theta = theta - lr * gradients


    return theta



data = fetch_california_housing()

X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# feature scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# training using gradient descent

theta = gradient_descent(X_train, y_train, lr=0.01, itr=2000)


# Predict

X_test_b = np.c_[np.ones(len(X_test)), X_test]

y_pred_gd = X_test_b.dot(theta)


# Train using Scikit learn

model = LinearRegression()

model.fit(X_train, y_train)

y_pred_sk = model.predict(X_test)


print("California Housing "
      "(GD):", mean_squared_error(y_test, y_pred_gd))



