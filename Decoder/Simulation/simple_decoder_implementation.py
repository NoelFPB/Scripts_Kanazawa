import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Define the dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 1, 2, 3])  # Outputs as class labels
y_encoded = to_categorical(y, num_classes=4)  # One-hot encode the outputs

# Build the model
model = Sequential([
    Dense(4, input_dim=2, activation='softmax')  # 2 inputs, 4 outputs
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_encoded, epochs=100, verbose=1)

# Test the model
print("Testing:")
for input_data in X:
    prediction = model.predict(np.array([input_data]))
    print(f"Input: {input_data}, Prediction Array: {prediction}, Predicted Output: {np.argmax(prediction)}")
