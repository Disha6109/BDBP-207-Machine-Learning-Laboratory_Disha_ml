
theta = [2, 3, 3]


X = [
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
]


result = [] # storing values of X0


for row in X:
    value = row[0]*theta[0] + row[1]*theta[1] + row[2]*theta[2]
    result.append(value)


print("Result of Xθ:")
for value in result:
    print(value)