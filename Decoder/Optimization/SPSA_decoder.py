import serial
import time
import pyvisa
import json
import random

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
heater_values = {i: 0.0 for i in range(40)}
fixed_first_layer = list(range(33, 40))
input_heaters = [36, 37]
input_combinations = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
modifiable_heaters = [i for i in range(33) if i not in input_heaters]

# Set fixed first layer
for heater in fixed_first_layer:
    heater_values[heater] = 0.01

# Voltage options - simplified set for faster convergence
voltage_options = [0.1, 2.5, 4.9]

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
    """Evaluate a single configuration for decoder behavior"""
    total_score = 0
    results = []
    
    # Expected highest output for each input combination
    expected_outputs = {
        (0.1, 0.1): 0,  # Output1 should be highest
        (0.1, 4.9): 1,  # Output2 should be highest
        (4.9, 0.1): 2,  # Output3 should be highest
        (4.9, 4.9): 3   # Output4 should be highest
    }
    
    for input_state in input_combinations:
        current_config = config.copy()
        current_config[36] = input_state[0]
        current_config[37] = input_state[1]
        
        send_heater_values(ser, current_config)
        time.sleep(0.25)
        
        outputs = measure_outputs()
        if None in outputs:
            return -1000, []
        
        # Find highest output
        max_output = max(outputs)
        actual_highest = outputs.index(max_output)
        expected_highest = expected_outputs[input_state]
        
        # Score based on correct output being highest
        if actual_highest == expected_highest:
            # Additional points for separation
            other_outputs = outputs.copy()
            other_outputs.pop(actual_highest)
            separation = max_output - max(other_outputs)
            total_score += 10 + min(separation * 5, 10)  # Up to 10 bonus points for separation
        
        results.append({
            'inputs': input_state,
            'outputs': outputs,
            'expected_highest': expected_highest,
            'actual_highest': actual_highest,
            'separation': max_output - max(other_outputs) if actual_highest == expected_highest else 0
        })
    
    return total_score, results

def spsa_gradient(ser, current_config, delta, voltage_options):
    """Estimate the gradient using SPSA"""
    delta_vector = {h: random.choice([-1, 1]) for h in modifiable_heaters}
    
    # Perturb configurations
    config_plus = current_config.copy()
    config_minus = current_config.copy()
    
    for heater in modifiable_heaters:
        config_plus[str(heater)] = max(voltage_options[0], min(voltage_options[-1], current_config[str(heater)] + delta * delta_vector[heater]))
        config_minus[str(heater)] = max(voltage_options[0], min(voltage_options[-1], current_config[str(heater)] - delta * delta_vector[heater]))
    
    # Evaluate configurations
    score_plus, _ = evaluate_configuration(ser, config_plus)
    score_minus, _ = evaluate_configuration(ser, config_minus)
    
    # Estimate gradient
    gradient = {}
    for heater in modifiable_heaters:
        gradient[heater] = (score_plus - score_minus) / (2 * delta * delta_vector[heater])
    
    return gradient

def spsa_optimization(ser, iterations=50, delta=0.1, learning_rate=0.05):
    """Optimize heater configuration using SPSA"""
    # Initialize with random configuration
    best_config = {str(i): random.choice(voltage_options) for i in modifiable_heaters}
    for heater in fixed_first_layer:
        best_config[str(heater)] = 0.01
    
    best_score, _ = evaluate_configuration(ser, best_config)
    print(f"Initial score: {best_score}")
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}/{iterations}")
        
        # Compute gradient
        gradient = spsa_gradient(ser, best_config, delta, voltage_options)
        
        # Update configuration
        for heater in modifiable_heaters:
            current_value = best_config[str(heater)]
            best_config[str(heater)] = max(
                voltage_options[0], 
                min(voltage_options[-1], current_value - learning_rate * gradient[heater])
            )
        
        # Evaluate updated configuration
        current_score, _ = evaluate_configuration(ser, best_config)
        print(f"Current score: {current_score}")
        
        if current_score > best_score:
            best_score = current_score
            print("Improved!")
        else:
            print("No improvement.")
    
    return best_config, best_score

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(2)
        
        print("Starting SPSA optimization...")
        best_config, best_score = spsa_optimization(ser)
        
        print("\nOptimization Complete!")
        print(f"Best score: {best_score}")
        
        # Print final heater configuration
        print("\nFinal Heater Configuration:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f}V")
        
        # Save configuration
        with open("best_spsa_configuration.json", 'w') as f:
            json.dump(best_config, f, indent=4)
        
        # Test final configuration with detailed analysis
        print("\nTesting final configuration:")
        for input_state in input_combinations:
            current_config = best_config.copy()
            current_config[36] = input_state[0]
            current_config[37] = input_state[1]
            
            send_heater_values(ser, current_config)
            time.sleep(2)
            outputs = measure_outputs()
            
            max_output = max(outputs)
            max_index = outputs.index(max_output)
            
            print(f"\nInputs (A, B): {input_state}")
            print(f"Outputs: {outputs}")
            print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
            other_outputs = outputs.copy()
            other_outputs.pop(max_index)
            print(f"Separation from next highest: {(max_output - max(other_outputs)):.4f}V")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        ser.close()
        scope.close()

if __name__ == "__main__":
    main()
