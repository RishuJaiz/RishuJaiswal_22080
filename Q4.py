def Input(rows, cols):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            element = int(input(f"Enter element at position ({i+1}, {j+1}): "))
            row.append(element)
        matrix.append(row)
    return matrix
def returnTranspose(mat):
    result=[[0 for i in range(len(mat))] for j in range(len(mat[0]))]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            result[i][j]=mat[j][i]
    return result

rows = int(input("Number of rows : "))
cols = int(input("Number of columns : "))
print("Enter elements for the matrix:")
input = Input(rows, cols)
TransMatrix = returnTranspose(input)

print("\nOriginal Matrix:")
for row in input:
    print(row)

print("\nTranspose of the given Matrix:")
for row in returnTransposeMatrix:
    print(row)
