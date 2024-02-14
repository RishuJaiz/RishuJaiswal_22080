def get_point():
    x = float(input("Enter the x-coordinate: "))
    y = float(input("Enter the y-coordinate: "))
    return (x, y)

print("Enter coordinates for the first point:")
point_a = get_point()

print("Enter coordinates for the second point:")
point_b = get_point()

def euclidean_distance(p1, p2):
    diff = [p1[i] - p2[i] for i in range(len(p1))]
    square_diff = [d ** 2 for d in diff]
    square_sum = sum(square_diff)
    return square_sum ** 0.5

def manhattan_distance(p1, p2):
    return sum(abs(p1[i] - p2[i]) for i in range(len(p1)))

dis1 = euclidean_distance(point_a, point_b)
dis2 = manhattan_distance(point_a, point_b)

print("Euclidean distance = ", dis1)
print("Manhattan distance = ", dis2)
