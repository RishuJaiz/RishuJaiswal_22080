import numpy as np
import matplotlib.pyplot as plt

# Generating 20 data points with random values between 1 and 10
np.random.seed(0)  # ensures reproducibility of random numbers
X = np.random.randint(1, 11, size=(20, 2))

print(X)

# Assign these points to two different classes
class0 = X[:10]
class1 = X[10:]

# Plot the training data
plt.figure(figsize=(8, 6))
plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0')
plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
plt.title('Scatter Plot of Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
