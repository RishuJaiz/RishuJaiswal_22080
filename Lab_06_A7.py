import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def derivative_sigmoid(x):
    return x * (1 - x)

# Function to train the neural network
def train_neural_network(inputs, targets, learning_rate, convergence_threshold, max_iterations=1000):


    # Initialize weights randomly
    np.random.seed(0)
    input_size = inputs.shape[1]
    hidden_size = 2
    output_size = 1
    weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
    weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

    for iteration in range(max_iterations):
        total_error = 0
        for i in range(len(inputs)):
            # Forward pass
            hidden_input = np.dot(inputs[i], weights_input_hidden)
            hidden_output = sigmoid(hidden_input)
            output_input = np.dot(hidden_output, weights_hidden_output)
            output = sigmoid(output_input)

            # Calculate error
            error = targets[i] - output
            total_error += error**2

            # Backpropagation
            output_delta = error * derivative_sigmoid(output)
            hidden_error = np.dot(output_delta, weights_hidden_output.T)
            hidden_delta = hidden_error * derivative_sigmoid(hidden_output)

            # Update weights
            weights_hidden_output += learning_rate * np.outer(hidden_output, output_delta)
            weights_input_hidden += learning_rate * np.outer(inputs[i], hidden_delta)

        # Check for convergence
        average_error = total_error / len(inputs)
        if average_error <= convergence_threshold:
            return weights_input_hidden, weights_hidden_output, iteration + 1

    return weights_input_hidden, weights_hidden_output, max_iterations

# Function to test the neural network
def test_neural_network(inputs, weights_input_hidden, weights_hidden_output):


    results = []
    for i in range(len(inputs)):
        # Forward pass
        hidden_input = np.dot(inputs[i], weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output)
        output = sigmoid(output_input)
        results.append((inputs[i], output))
    return results

def main():
    # Define the AND gate truth table
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [0], [0], [1]])

    # Hyperparameters
    learning_rate = 0.05
    convergence_threshold = 0.002

    # Train the neural network
    weights_input_hidden, weights_hidden_output, convergence_epoch = train_neural_network(inputs, targets, learning_rate, convergence_threshold)

    # Test the trained neural network
    results = test_neural_network(inputs, weights_input_hidden, weights_hidden_output)

    # Print results
    print("Input\tOutput")
    for input_data, output in results:
        print(f"{input_data}\t{output}")

    if convergence_epoch < 1000:
        print(f"Converged after {convergence_epoch} iterations.")
    else:
        print("Did not converge within 1000 iterations.")

if __name__ == "__main__":
    main()
