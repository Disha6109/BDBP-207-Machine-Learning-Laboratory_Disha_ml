# Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are three independent variables. Compute the gradient of y at a few points and print the values.
def linear_model(x1, x2, x3):
    return 2*x1 + 3*x2 + 3*x3 + 4

gradient = [2, 3, 3]

points = [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 4, 2)
]

for p in points:
    print("At point", p, "gradient =", gradient)