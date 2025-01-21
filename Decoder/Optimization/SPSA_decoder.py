import serial
import time
import pyvisa
import json
import numpy as np

# Serial port configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Initialize oscilloscope
rm = pyvisa.ResourceManager()
resources = rm.list_resources()
if not resources:
    raise Exception("No VISA resources found")
scope = rm.open_resource(resources[0])
scope.timeout = 5000

# Setup channels
channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
for channel in channels:
    scope.write(f':{channel}:DISPlay ON')
    scope.write(f':{channel}:SCALe 2')
    scope.write(f':{channel}:OFFSet -6')

# Heater configuration
fixed_first_layer = list(range(33, 40))
input_heaters = [36, 37]
input_combinations = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
modifiable_heaters = [i for i in range(33) if i not in input_heaters]

def send_heater_values(ser, config):
    """Send heater values via serial connection"""
    voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.01)

def measure_outputs():
    """Measure outputs from oscilloscope"""
    try:
        outputs = []
        for channel in range(1, 5):
            value = float(scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
            outputs.append(round(value, 5))
        return outputs
    except Exception as e:
        print(f"Measurement error: {e}")
        return [None] * 4

def evaluate_configuration(ser, config):
    """
    Evaluate a configuration based on how well it performs as a decoder.
    Returns a single scalar value representing performance.
    """
    outputs_by_input = []
    
    for input_state in input_combinations:
        current_config = config.copy()
        current_config[36] = input_state[0]
        current_config[37] = input_state[1]
        
        send_heater_values(ser, current_config)
        time.sleep(0.25)
        
        outputs = measure_outputs()
        if None in outputs:
            return float('-inf')  # Return negative infinity for failed measurements
            
        outputs_by_input.append(outputs)
    
    # Calculate loss based on desired behavior
    total_loss = 0
    
    # For each input combination, we want one output to be high and others low
    for idx, outputs in enumerate(outputs_by_input):
        desired_output = idx  # We want output i to be highest for input combination i
        
        # Calculate softmax of outputs
        exp_outputs = np.exp(outputs - np.max(outputs))  # Subtract max for numerical stability
        softmax = exp_outputs / exp_outputs.sum()
        
        # Cross-entropy loss
        loss = -np.log(softmax[desired_output] + 1e-10)  # Add small epsilon to prevent log(0)
        total_loss += loss
    
    return -total_loss  # Return negative loss as we want to maximize

def spsa_optimization(ser, max_iterations=50):
    """
    Optimize heater configuration using SPSA with adaptive step sizes
    """
    # Initialize parameters
    n_params = len(modifiable_heaters)
    theta = np.random.uniform(0, 5, n_params)  # Initial parameter values
    
    # SPSA parameters
    a = 0.1  # Initial step size
    c = 0.1  # Initial perturbation size
    A = max_iterations/10  # Stability constant
    alpha = 0.602  # Decay rate for step size
    gamma = 0.101  # Decay rate for perturbation
    
    best_score = float('-inf')
    best_theta = theta.copy()
    
    for k in range(max_iterations):
        # Calculate current step sizes
        ak = a / ((k + 1 + A) ** alpha)
        ck = c / ((k + 1) ** gamma)
        
        # Generate random perturbation vector
        delta = 2 * np.random.binomial(1, 0.5, n_params) - 1
        
        # Create parameter vectors for gradient approximation
        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta
        
        # Convert to configurations
        config_plus = {str(h): max(0, min(5, v)) for h, v in zip(modifiable_heaters, theta_plus)}
        config_minus = {str(h): max(0, min(5, v)) for h, v in zip(modifiable_heaters, theta_minus)}
        
        # Add fixed heaters
        for h in fixed_first_layer:
            config_plus[str(h)] = 0.01
            config_minus[str(h)] = 0.01
            
        # Evaluate both configurations
        y_plus = evaluate_configuration(ser, config_plus)
        y_minus = evaluate_configuration(ser, config_minus)
        
        # Calculate gradient approximation
        g_hat = (y_plus - y_minus) / (2 * ck * delta)
        
        # Update parameters
        theta = theta + ak * g_hat
        
        # Clip parameters to valid range [0, 5]
        theta = np.clip(theta, 0.1, 4.9)
        
        # Convert current theta to configuration and evaluate
        current_config = {str(h): v for h, v in zip(modifiable_heaters, theta)}
        for h in fixed_first_layer:
            current_config[str(h)] = 0.01
            
        current_score = evaluate_configuration(ser, current_config)
        
        # Update best solution if necessary
        if current_score > best_score:
            best_score = current_score
            best_theta = theta.copy()
            
        
        print(f"Iteration {k}: Score = {current_score:.2f}, Best = {best_score:.2f}")
    
    # Convert best solution to configuration
    best_config = {str(h): v for h, v in zip(modifiable_heaters, best_theta)}
    for h in fixed_first_layer:
        best_config[str(h)] = 0.01
        
    return best_config, best_score

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        print("Starting SPSA optimization...")
        best_config, best_score = spsa_optimization(ser)
        
        print("\nOptimization Complete!")
        print(f"Best score: {best_score}")
        
        # Save configuration
        with open("best_spsa_configuration.json", 'w') as f:
            json.dump(best_config, f, indent=4)
        
        # Test final configuration
        print("\nTesting final configuration:")
        for input_state in input_combinations:
            current_config = best_config.copy()
            current_config[36] = input_state[0]
            current_config[37] = input_state[1]
            
            send_heater_values(ser, current_config)
            time.sleep(0.25)
            outputs = measure_outputs()
            
            print(f"\nInputs (A, B): {input_state}")
            print(f"Outputs: {outputs}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        ser.close()
        scope.close()

if __name__ == "__main__":
    main()