A = [[1, 2, 3],
     [4, 5, 6]]

transpose_t= [[0,0],
              [0,0],
              [0,0]]

for i in range(len(A)):
    for j in range(len(A[i])):
        transpose_t[j][i] = A[i][j]

print(transpose_t)














