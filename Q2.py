def multiply_matrices(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    if cols_a != rows_b:
        return None  # Matrices are not multipliable

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result

def user_input_matrix(rows, cols):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            element = int(input(f"Enter element at position ({i + 1}, {j + 1}): "))
            row.append(element)
        matrix.append(row)
    return matrix

rows_a = int(input("Enter the number of rows for matrix A: "))
cols_a = int(input("Enter the number of columns for matrix A: "))
rows_b = int(input("Enter the number of rows for matrix B: "))
cols_b = int(input("Enter the number of columns for matrix B: "))
print("\nEnter elements for matrix A:")
matrix_A = user_input_matrix(rows_a, cols_a)
print("\nEnter elements for matrix B:")
matrix_B = user_input_matrix(rows_b, cols_b)
result_matrix = multiply_matrices(matrix_A, matrix_B)
if result_matrix is None:
    print("\nError: Matrices are not multipliable")
else:
    print("\nMatrix A:")
    for row in matrix_A:
        print(row)

    print("\nMatrix B:")
    for row in matrix_B:
        print(row)

    print("\nResultant Matrix AB:")
    for row in result_matrix:
        print(row)
