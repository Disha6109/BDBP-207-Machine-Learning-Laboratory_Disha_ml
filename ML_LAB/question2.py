import numpy as np

import matplotlib.pyplot as plt

def sigmoid(z):

    return 1/(1+np.exp(-z))


z = np.linspace(-10,10,100)

plt.plot(z,sigmoid(z))

plt.xlabel("z")

plt.ylabel("Sigmoid(z)")

plt.title("Sigmoid Function")

plt.grid()

plt.show()