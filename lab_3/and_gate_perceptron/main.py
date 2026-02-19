import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# 2. Hyperparameters
lr = 0.1
threshold = 0.5
weights = np.random.rand(2) 

print(f"Starting weights: {weights}")

# 3. Training Loop
converged = False
epoch = 0

while not converged:
    converged = True
    epoch += 1
    print(f"\n--- Epoch {epoch} ---")

    for x_input, target in zip(X, y):
        # Calculate weighted sum (Dot Product)
        weighted_sum = np.dot(x_input, weights)

        # Activation (Step Function)
        prediction = 1 if weighted_sum >= threshold else 0

        # Update weights if there's an error
        error = target - prediction
        if error != 0:
            weights += lr * error * x_input
            converged = False
            print(
                f"Input: {x_input} | Target: {target} | Pred: {prediction} | New Weights: {weights}"
            )
        else:
            print(f"Pass for the input :{x_input} with prediction:{prediction}")

print(f"\nTraining complete in {epoch} epochs.")
print(f"Final Weights: {weights}")
