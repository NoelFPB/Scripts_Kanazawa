import numpy as np
import cmath
import matplotlib.pyplot as plt

# Define 2-bit decoder dataset
inputs = np.array([
    [0, 0],  # 00
    [0, 1],  # 01
    [1, 0],  # 10
    [1, 1]   # 11
], dtype=np.float32)

targets = np.array([
    [1, 0, 0, 0],  # 00 -> [1, 0, 0, 0]
    [0, 1, 0, 0],  # 01 -> [0, 1, 0, 0]
    [0, 0, 1, 0],  # 10 -> [0, 0, 1, 0]
    [0, 0, 0, 1]   # 11 -> [0, 0, 0, 1]
], dtype=np.float32)

# Convert 2-bit input to a 7-dimensional input vector
def GetSevenInput(bits):
    return [bits[0], bits[1], 0, 0, 0, 0, 0]

# Quantum-inspired transformation
def yout(x1, x2, v1, v2):
    y1 = (x1 * cmath.exp(1j * 2 * cmath.pi * v1) + 1j * (x2 * cmath.exp(1j * 2 * cmath.pi * v2))) / cmath.sqrt(2)
    y2 = (1j * x1 * cmath.exp(1j * 2 * cmath.pi * v1) + (x2 * cmath.exp(1j * 2 * cmath.pi * v2))) / cmath.sqrt(2)
    return y1, y2

# Circle function with 4 output values
def Circle(x, v):
    x = GetSevenInput(x)
    # Layer 1
    y1_1, y1_2 = yout(x[0], x[1], v[0], v[1])
    # Layer 2
    y2_1, y2_2 = yout(x[2], x[3], v[2], v[3])
    # Layer 3
    y3_1, y3_2 = yout(x[4], x[5], v[4], v[5])
    # Combine outputs for final output
    y_out1, y_out2 = yout(y1_1, y2_1, v[6], v[6])
    y_out3, y_out4 = yout(y2_2, y3_2, v[6], v[6])
    return [abs(y_out1)**2, abs(y_out2)**2, abs(y_out3)**2, abs(y_out4)**2]

# Compute Mean Squared Error (MSE)
def GetAverageMSE(output_matrix, targets):
    mse = 0
    for i in range(len(targets)):
        mse += np.sum((output_matrix[i] - targets[i])**2)
    return mse / len(targets)

# Training parameters
N_Circle = 8      # Number of layers
N_weight = 7      # Number of weights per layer
step = 0.1       # Step size for weight adjustment
N_changeweight = 200  # Number of weight adjustments
changed_weight = np.random.rand(N_Circle, N_weight)  # Initialize weights
MSEav_Array = []  # Store MSE over iterations

# Training loop
for iteration in range(1000):  # Number of iterations
    for _circle in range(N_Circle):  # Loop through each layer
        for _weight in range(N_weight):  # Loop through each weight
            for i in range(N_changeweight):  # Optimize each weight
                plusMat = changed_weight.copy()
                plusMat[_circle][_weight] += step  # Increase weight
                minusMat = changed_weight.copy()
                minusMat[_circle][_weight] -= step  # Decrease weight

                # Compute outputs
                output_plus = np.array([Circle(x, plusMat[_circle]) for x in inputs])
                output_minus = np.array([Circle(x, minusMat[_circle]) for x in inputs])

                # Compute MSE
                plusMSE = GetAverageMSE(output_plus, targets)
                minusMSE = GetAverageMSE(output_minus, targets)

                # Choose the direction that reduces MSE
                if plusMSE < minusMSE:
                    changed_weight = plusMat.copy()
                    MSEav_Array.append(plusMSE)
                else:
                    changed_weight = minusMat.copy()
                    MSEav_Array.append(minusMSE)

# Plot MSE
plt.plot(MSEav_Array)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Training Performance')
plt.show()

# Test the trained decoder
for i, x in enumerate(inputs):
    output = Circle(x, changed_weight[-1])  # Pass the input through the last trained layer
    output = [float(o) for o in output]  # Convert numpy.float32 to Python float
    predicted = np.argmax(output)  # Get the index of the largest output
    print(f"Input: {x}, Output: {output}, Predicted: {predicted}, Target: {np.argmax(targets[i])}")
