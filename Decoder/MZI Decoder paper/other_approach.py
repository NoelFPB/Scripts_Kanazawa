import numpy as np
from scipy.linalg import expm

# Generate a random unitary matrix for each layer
def random_unitary_matrix(size):
    # Generate a random Hermitian matrix
    H = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    H = (H + H.conj().T) / 2
    # Use the matrix exponential to create a unitary matrix
    U = expm(1j * H)  # Unitary matrix: exp(iH)
    return U

def optical_propagation(input_vector, layers):
    """
    Propagate an input vector through the optical layers.
    Args:
        input_vector (ndarray): Input light state (8-dimensional complex vector).
        layers (list): List of unitary matrices (one per layer).
    Returns:
        ndarray: Final output vector after propagation.
    """
    state = input_vector
    for layer in layers:
        state = np.dot(layer, state)  # Apply unitary transformation
    return state

def loss_function(predicted, target):
    """
    Compute the mean squared error (MSE) between predicted and target outputs.
    Args:
        predicted (ndarray): Predicted output state.
        target (ndarray): Target output state.
    Returns:
        float: MSE value.
    """
    return np.mean(np.abs(predicted - target)**2)

from scipy.optimize import minimize

def train_optical_circuit(inputs, targets, layers, max_iter=100):
    """
    Train the optical circuit to minimize the loss between propagated outputs and targets.
    Args:
        inputs (list): List of input light states (8-dimensional complex vectors).
        targets (list): List of target output states.
        layers (list): Initial unitary matrices representing the layers.
        max_iter (int): Maximum number of optimization iterations.
    Returns:
        list: Trained unitary matrices.
    """
    # Flatten parameters for optimization
    params = []
    for layer in layers:
        params.extend(layer.flatten().view(np.float64))  # Convert complex to real
    
    def objective(flat_params):
        # Reconstruct layers from flat_params
        idx = 0
        for i in range(len(layers)):
            size = layers[i].shape[0]
            flat_layer = flat_params[idx:idx + 2 * size**2].view(np.complex128)
            layers[i] = flat_layer.reshape((size, size))
            idx += 2 * size**2

        # Compute total loss
        total_loss = 0
        for input_vector, target_vector in zip(inputs, targets):
            predicted = optical_propagation(input_vector, layers)
            total_loss += loss_function(predicted, target_vector)
        return total_loss

    # Optimize using SciPy's L-BFGS-B optimizer
    result = minimize(objective, params, method='L-BFGS-B', options={'maxiter': max_iter})
    return layers

# Generate random inputs and targets for testing
num_samples = 10
input_vectors = [np.random.randn(8) + 1j * np.random.randn(8) for _ in range(num_samples)]
target_vectors = [np.random.randn(8) + 1j * np.random.randn(8) for _ in range(num_samples)]

# Initialize random layers
num_layers = 7
layers = [random_unitary_matrix(8) for _ in range(num_layers)]

# Train the circuit
trained_layers = train_optical_circuit(input_vectors, target_vectors, layers, max_iter=200)

# Test the circuit
for input_vector, target_vector in zip(input_vectors, target_vectors):
    predicted_output = optical_propagation(input_vector, trained_layers)
    print(f"Target: {target_vector}")
    print(f"Predicted: {predicted_output}")
    print(f"Loss: {loss_function(predicted_output, target_vector)}")
