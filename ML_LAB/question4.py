import matplotlib.pyplot as plt
import math

x = []
y = []

mean = 0
sigma = 15

start = -100
stop = 100
num = 100
step = (stop - start) / (num - 1)

for i in range(num):
    val = start + i * step
    x.append(val)
    exponent = -((val - mean)**2) / (2*sigma*sigma)
    y.append((1/(sigma*math.sqrt(2*math.pi))) * math.exp(exponent))

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Gaussian PDF")
plt.show()