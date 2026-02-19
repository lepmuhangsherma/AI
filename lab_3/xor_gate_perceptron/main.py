import numpy as np


# 1. Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 2. Input Data (XOR Input) and Expected Output
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# 3. Initialize weights and biases with random values
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Weights from Input to Hidden layer
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))

# Weights from Hidden to Output layer
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))

lr = 0.5  # Learning rate
epochs = 10000  # Number of iterations

# 4. Training Loop
for _ in range(epochs):
    # --- Forward Propagation ---
    hidden_layer_activation = np.dot(inputs, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)

    # --- Backpropagation ---
    # Calculate error at output
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    # Calculate error at hidden layer
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # --- Updating Weights and Biases ---
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

# 5. Final Output
print("Final Predicted Output after 10,000 epochs:")
print(predicted_output.round())
