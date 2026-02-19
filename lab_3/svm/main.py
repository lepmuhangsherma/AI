import numpy as np
from sklearn import svm

# 1. Sample Data: [Feature 1, Feature 2]
# Let's say: [Height, Weight]
X = np.array([[170, 70], [180, 80], [150, 50], [160, 60], [175, 75], [155, 55]])

# 2. Labels: 0 = 'Small', 1 = 'Large'
y = np.array([1, 1, 0, 0, 1, 0])

# 3. Create the SVM model
# 'linear' kernel means we want a straight line boundary
model = svm.SVC(kernel="linear")

# 4. Train the model
model.fit(X, y)

# 5. Predict for a new person [168, 68]
prediction = model.predict([[168, 68]]) # type: ignore

# 6. Output the result
label = "Large" if prediction[0] == 1 else "Small"
print(f"The model classifies [168, 68] as: {label}")

# See the 'Support Vectors'
print("\nSupport Vectors (the most important points):")
print(model.support_vectors_)
