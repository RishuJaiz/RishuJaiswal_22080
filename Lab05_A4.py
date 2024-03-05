import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Generate random training data
np.random.seed(0)
training_data = np.random.randint(1, 11, size=(20, 2))  # 20 points between 1 and 10 for X and Y
labels = np.random.randint(2, size=20)  #(0 or 1) labels

# Generate test set data
x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
test_data = np.array([[x, y] for x in x_values for y in y_values])

# Initialize the kNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(training_data, labels)

# Predict the classes for the test data
predicted_classes = knn.predict(test_data)

# Separate test data based on predicted classes
class0 = test_data[predicted_classes == 0]
class1 = test_data[predicted_classes == 1]

# Plot the test data with predicted classes
plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0')
plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Test Data with Predicted Classes')
plt.legend()
plt.show()
