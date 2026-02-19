import numpy as np

# NOT Gate Truth Table
inputs = np.array([0, 1])
targets = np.array([1, 0])

# Initialize
weight = -0.1
bias = 0.1
lr = 0.1
epochs = 20

print("Training NOT Gate Perceptron...")

for epoch in range(epochs):
    for i in range(len(inputs)):
        # Calculate Weighted Sum
        z = inputs[i] * weight + bias

        # Heaviside Step Function (Activation)
        prediction = 1 if z >= 0 else 0

        # Calculate Error
        error = targets[i] - prediction

        # Update Weight and Bias
        weight += lr * error * inputs[i]
        bias += lr * error

# Test the results
print("\nFinal Weight:", round(weight, 2))
print("Final Bias:", round(bias, 2))
print("-" * 20)

for i in range(len(inputs)):
    z = inputs[i] * weight + bias
    out = 1 if z >= 0 else 0
    print(f"Input: {inputs[i]} | Target: {targets[i]} | Predicted: {out}")
