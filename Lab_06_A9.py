import numpy as np
import matplotlib.pyplot as plt

# Define the AND gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Output labels for two output nodes: [1, 0] represents 0, and [0, 1] represents 1
y = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])

# Define initial weights
W = np.array([[10, 0.2, -0.75], [10, 0.2, -0.75]])  # Two sets of weights for two output nodes

# Learning rate
alpha = 0.05

# Step activation function for each output node
def step_function(x):
    return np.array([1 if val >= 0 else 0 for val in x])

# Bipolar step activation function
def bipolar_step_function(x):
    return np.array([-1 if val < 0 else 1 for val in x])

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Perceptron function for two output nodes
def perceptron(x, weights, activation_function):
    # Compute activations for each output node
    activations = np.dot(x, weights[:, 1:]) + weights[:, 0]
    # Apply the specified activation function to each activation
    return activation_function(activations)

# Training the perceptron for two output nodes
def train_perceptron(X, y, weights, activation_function):
    errors = [calculate_error(X, y, weights)]  # Initialize errors with initial error
    epochs = 0   # Count of epochs
    convergence_error = 0.002  # Error threshold for convergence
    max_epochs = 1000  # Maximum number of epochs to prevent infinite looping

    # Training loop
    while True:
        # Check for convergence or max epochs
        if errors[-1] <= convergence_error or epochs >= max_epochs:
            break
        # Iterate over each input-output pair
        for i in range(len(X)):
            # Get predictions for the current input using the perceptron function
            predictions = perceptron(X[i], weights, activation_function)
            # Update weights based on prediction errors
            for j in range(len(weights)):
                error = y[i][j] - predictions[j]
                weights[j, 1:] += alpha * error * X[i]
                weights[j, 0] += alpha * error
        # Increment epoch count
        epochs += 1
        # Append the error after each epoch
        errors.append(calculate_error(X, y, weights))

    return epochs, errors, weights

# Calculate error for two output nodes
def calculate_error(X, y, weights):
    errors = 0
    # Iterate over each input-output pair
    for i in range(len(X)):
        # Get predictions for the current input using the perceptron function
        predictions = perceptron(X[i], weights, step_function)
        # Compute squared error for each output node and sum them up
        errors += np.sum((y[i] - predictions) ** 2)
    # Average the total error over all input-output pairs
    return errors / (len(X) * len(y[0]))

# Train the perceptron using the step activation function
epochs_step, errors_step, final_weights_step = train_perceptron(X, y, W, step_function)

# Plotting the error for step activation function
plt.plot(range(epochs_step + 1), errors_step)  # Add 1 to include initial error
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs (Step Activation Function)')
plt.show()

print("Number of epochs needed for convergence (Step Activation Function):", epochs_step)
print("Final weights (Step Activation Function):", final_weights_step)

# Train the perceptron using the bipolar step activation function
epochs_bipolar, errors_bipolar, final_weights_bipolar = train_perceptron(X, y, W, bipolar_step_function)

# Plotting the error for bipolar step activation function
plt.plot(range(epochs_bipolar + 1), errors_bipolar)  # Add 1 to include initial error
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs (Bipolar Step Activation Function)')
plt.show()

print("Number of epochs needed for convergence (Bipolar Step Activation Function):", epochs_bipolar)
print("Final weights (Bipolar Step Activation Function):", final_weights_bipolar)

# Train the perceptron using the sigmoid activation function
epochs_sigmoid, errors_sigmoid, final_weights_sigmoid = train_perceptron(X, y, W, sigmoid)

# Plotting the error for sigmoid activation function
plt.plot(range(epochs_sigmoid + 1), errors_sigmoid)  # Add 1 to include initial error
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs (Sigmoid Activation Function)')
plt.show()

print("Number of epochs needed for convergence (Sigmoid Activation Function):", epochs_sigmoid)
print("Final weights (Sigmoid Activation Function):", final_weights_sigmoid)

# Train the perceptron using the ReLU activation function
epochs_relu, errors_relu, final_weights_relu = train_perceptron(X, y, W, relu)

# Plotting the error for ReLU activation function
plt.plot(range(epochs_relu + 1), errors_relu)  # Add 1 to include initial error
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs (ReLU Activation Function)')
plt.show()

print("Number of epochs needed for convergence (ReLU Activation Function):", epochs_relu)
print("Final weights (ReLU Activation Function):", final_weights_relu)
