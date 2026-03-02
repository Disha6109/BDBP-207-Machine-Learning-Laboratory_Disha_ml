import numpy as np

X = np.array([[10], [11], [12], [13], [14]])

y = np.array([2, 4, 6, 8, 10])

X = np.c_[np.ones(X.shape[0]), X]

theta = np.zeros(X.shape[1])

learning_rate = 0.01

batch_set = 2


m = len(y)

no_of_steps= 1000


for i in range(no_of_steps):

    numbers = np.random.permutation(m)

    Xb = X[numbers[:batch_set]]

    yb = y[numbers[:batch_set]]

    predictions = Xb @ theta

    error = predictions - yb


    gradient = (Xb.T @ error) / batch_set

    theta = theta - learning_rate * gradient


print("Final value of Theta:", theta)

