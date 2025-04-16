# This approach works well

import serial
import time
import pyvisa
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

# Setup only Channel 1 for AND gate output measurement
scope.write(':CHANnel1:DISPlay ON')
scope.write(':CHANnel1:SCALe 2')
scope.write(':CHANnel1:OFFSet -6')

# Turn off other channels as we don't need them
for channel_num in range(2, 5):
    scope.write(f':CHANnel{channel_num}:DISPlay OFF')

# Heater configuration
heater_values = {i: 0.0 for i in range(40)}
fixed_first_layer = list(range(33, 40))
input_heaters = [36, 37]  # These are our AND gate inputs
input_combinations = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
modifiable_heaters = [i for i in range(33) if i not in input_heaters]

# Set fixed first layer
for heater in fixed_first_layer:
    heater_values[heater] = 0.01

# Voltage options - simplified set for faster convergence
# For AND gate, including more options in the lower range might be helpful
voltage_options = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.9]

def send_heater_values(ser, config):
    """Send heater values via serial connection"""
    voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()

def measure_output():
    """Measure just the AND gate output from oscilloscope (Channel 1)"""
    try:
        value = float(scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel1'))
        return round(value, 5)
    except Exception as e:
        print(f"Measurement error: {e}")
        return None

def evaluate_configuration_for_and_gate(ser, config):
    """Evaluate a single configuration for AND gate behavior"""
    total_score = 0
    results = []
    
    # For AND gate, we want output high (1) only when both inputs are high
    expected_outputs = {
        (0.1, 0.1): 0,  # Both low -> output should be low
        (0.1, 4.9): 0,  # A low, B high -> output should be low
        (4.9, 0.1): 0,  # A high, B low -> output should be low
        (4.9, 4.9): 1   # Both high -> output should be high
    }
    
    for input_state in input_combinations:
        current_config = config.copy()
        current_config[36] = input_state[0]
        current_config[37] = input_state[1]
        
        send_heater_values(ser, current_config)
        time.sleep(0.25)
        
        and_output = measure_output()
        if and_output is None:
            return -1000, []
        
        # For AND gate behavior:
        # - When both inputs are high (4.9, 4.9), output should be high
        # - For all other input combinations, output should be low
        if input_state == (4.9, 4.9):
            # We want high output for both inputs high
            # Score based on how high the output is (higher is better)
            if and_output > 4.0:  # Threshold for "high"
                # Reward higher voltage for the high state
                total_score += 20 + min(and_output * 5, 20)
            else:
                # Penalize if not high enough
                total_score -= 10
        else:
            # For all other input combinations, output should be low
            # Score based on how low the output is (lower is better)
            if and_output < 1.0:  # Threshold for "low"
                # Reward lower voltage for the low state
                total_score += 10 + min((1.0 - and_output) * 5, 10)
            else:
                # Penalize if not low enough
                total_score -= 20
        
        results.append({
            'inputs': input_state,
            'output': and_output,
            'expected': 'high' if expected_outputs[input_state] == 1 else 'low',
            'actual': 'high' if and_output > 2.0 else 'low'
        })
    
    return total_score, results

def hill_climbing_optimization(ser, iterations=20):
    """Simple hill climbing optimization for AND gate behavior"""
    # Initialize with random configuration
    best_config = {i: random.choice(voltage_options) for i in modifiable_heaters}
    for heater in fixed_first_layer:
        best_config[heater] = 0.01
    
    best_score, _ = evaluate_configuration_for_and_gate(ser, best_config)
    print(f"Initial score: {best_score}")
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}/{iterations}")
        improved = False
        
        # Try to improve each heater
        for heater in modifiable_heaters:
            current_value = best_config[heater]
            
            # Try different values
            for new_value in voltage_options:
                if new_value != current_value:
                    test_config = best_config.copy()
                    test_config[heater] = new_value
                    
                    score, results = evaluate_configuration_for_and_gate(ser, test_config)
                    
                    if score > best_score:
                        best_score = score
                        best_config = test_config.copy()
                        improved = True
                        print(f"Improved score: {best_score}")
                        break
            
            if improved:
                break
        
        if not improved:
            print("No improvement found in this iteration")
            break
    
    return best_config, best_score

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(0.5)
        
        print("Starting AND gate optimization...")
        best_config, best_score = hill_climbing_optimization(ser)
        
        print("\nOptimization Complete!")
        print(f"Best score: {best_score}")
        
        # Print final heater configuration
        print("\nFinal Heater Configuration:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f}V")
        
        # Test final configuration with detailed analysis
        print("\nTesting final AND gate configuration:")
        for input_state in input_combinations:
            current_config = best_config.copy()
            current_config[36] = input_state[0]
            current_config[37] = input_state[1]
            
            send_heater_values(ser, current_config)
            time.sleep(0.25)
            output_value = measure_output()
            
            print(f"\nInputs (A, B): {input_state}")
            print(f"AND Output (Channel 1): {output_value:.4f}V")
            print(f"Output is: {'HIGH' if output_value > 2.0 else 'LOW'}")
            print(f"Expected: {'HIGH' if input_state == (4.9, 4.9) else 'LOW'}")
            print(f"Correct: {'Yes' if ((output_value > 2.0) == (input_state == (4.9, 4.9))) else 'No'}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        ser.close()
        scope.close()

if __name__ == "__main__":
    main()