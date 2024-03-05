import numpy as np
import matplotlib.pyplot as plt

def perceptron(x, w, b):
    """
    This function implements a perceptron with a step activation function.
    Args:
      x: A list of input values.
      w: A list of weights for the perceptron.
      b: The bias of the perceptron.
    Returns:
      The output of the perceptron.
    """
    return np.where(np.dot(x, w) + b >= 0, 1, 0)

def train_perceptron(inputs, expected_outputs, w, b, learning_rate=0.05, max_epochs=1000, convergence_threshold=0.002):
    """
    This function trains the perceptron using the provided inputs and expected outputs.
    Args:
      inputs: The training data inputs.
      expected_outputs: The expected outputs corresponding to the inputs.
      w: Initial weights of the perceptron.
      b: Initial bias of the perceptron.
      learning_rate: The learning rate for updating weights and bias (default is 0.05).
      max_epochs: Maximum number of epochs for training (default is 1000).
      convergence_threshold: The threshold for considering convergence (default is 0.002).
    Returns:
      w: Final weights of the perceptron.
      b: Final bias of the perceptron.
      errors: List of errors accumulated over epochs.
      epochs: List of epochs.
    """
    errors = []  # Initialize list to store errors
    epochs = []  # Initialize list to store epochs

    for epoch in range(max_epochs):
        epoch_error = 0  # Initialize error for the epoch
        
        for i in range(len(inputs)):
            actual_output = perceptron(inputs[i], w, b)
            error = expected_outputs[i] - actual_output
            epoch_error += abs(error)
            w += learning_rate * error * inputs[i]
            b += learning_rate * error
        
        errors.append(epoch_error)
        epochs.append(epoch)

        if epoch_error <= convergence_threshold:
            break

    return w, b, errors, epochs

# Define the training data (inputs and expected outputs for the AND gate).
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = np.array([0, 0, 0, 1])

# Initialize the weights and bias with the correct number of weights.
initial_weights = np.array([10, 0.2])  # Two weights corresponding to the two inputs.
initial_bias = 0

# Train the perceptron
final_weights, final_bias, errors, epochs = train_perceptron(inputs, expected_outputs, initial_weights, initial_bias)

# Print the final weights and bias.
print("Final weights:", final_weights)
print("Final bias:", final_bias)
print("Number of epochs:", len(epochs))

# Plot the epochs vs. errors.
plt.plot(epochs, errors)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Error vs. Epochs")
plt.grid(True)
plt.show()
