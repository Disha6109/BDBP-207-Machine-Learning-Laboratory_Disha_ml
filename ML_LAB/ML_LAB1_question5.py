import matplotlib.pyplot as plt


x_values = []
y_values = []


for x in range(-10, 11):
    x_values.append(x)
    y_values.append(x * x)


plt.plot(x_values, y_values)


plt.title("First Parabola (y = x²)")
plt.xlabel("x value")
plt.ylabel("y value (x_squared)")
plt.grid(True)

plt.show()


print("The slope is:")
for x in [-5, 0, 5]:
    slope = 2 * x
    print("At x =", x, "the slope is", slope)
