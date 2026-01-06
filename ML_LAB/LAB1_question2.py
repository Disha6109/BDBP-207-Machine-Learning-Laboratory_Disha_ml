import matplotlib.pyplot as plt

x1 = []
y = []

start = -100
stop = 100
num = 100

step = (stop - start) / (num - 1)

for i in range(num):
    x = start + i * step
    x1.append(x)
    y.append(2*x + 3)

plt.plot(x1, y)
plt.xlabel('x1')
plt.ylabel('y')
plt.title("y = 2x1 + 3")
plt.show()
