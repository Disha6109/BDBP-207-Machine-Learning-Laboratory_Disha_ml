import numpy as np

import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-10, 10, 100)

plt.plot(z, sigmoid(z))

plt.xlabel("z")

plt.ylabel("Sigmoid(z)")

plt.title("Sigmoid Function")

plt.grid()

plt.show()

#question3
def sigmoid_derivative(z):

    s = sigmoid(z)

    return s*(1-s)


plt.plot(z,sigmoid_derivative(z))

plt.xlabel("z")

plt.ylabel("Sigmoid'(z)")

plt.title("Derivative of Sigmoid")

plt.grid()

plt.show()

#question4

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


data = load_breast_cancer()


X_train, X_test, y_train, y_test = train_test_split(

    data.data, data.target, test_size=0.2, random_state=40

)


model = LogisticRegression(max_iter=10000)

model.fit(X_train,y_train)


y_pred = model.predict(X_test)


print("Accuracy:",accuracy_score(y_test,y_pred))
