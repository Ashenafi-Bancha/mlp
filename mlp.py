
"""
Group Members:
- Ashenafi Bancha-----UGR/1796/15
- Elham Jemal---------UGR/1757/14
- Feruza Hassen-------UGR/6423/15
- Ihsan Jemal---------UGR/9433/15
"""




import numpy as np

# Activation function and derivative

def sigmoid(z):
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    """
    Derivative of sigmoid with respect to z
    Assumes 'a' is already sigmoid(z)
    """
    return a * (1 - a)

# XOR Dataset

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# Set random seed for reproducibility

np.random.seed(42)

# Network Architecture

input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1

# Weight and Bias Initialization

weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size)
bias_hidden = np.random.rand(1, hidden_layer_size)

weights_hidden_output = np.random.rand(hidden_layer_size, output_layer_size)
bias_output = np.random.rand(1, output_layer_size)

# Training Parameters

learning_rate = 0.1
epochs = 10000

# Training Loop

for epoch in range(epochs):

    # ---- Forward Propagation ----
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_input)

    # ---- Loss Calculation (MSE) ----
    error = y - predicted_output
    loss = np.mean(error ** 2)

    # ---- Backpropagation ----
    output_delta = error * sigmoid_derivative(predicted_output)

    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # ---- Weight and Bias Updates ----
    weights_hidden_output += np.dot(hidden_output.T, output_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # ---- Optional: Print loss occasionally ----
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Final Predictions

print("\nFinal Predictions:")
for i in range(len(X)):
    print(f"{X[i]} -> {predicted_output[i][0]:.4f}")
