import serial
import time
import pyvisa
import random
import json
import pandas as pd
import csv
import numpy as np
from scipy.optimize import differential_evolution

# Serial port configuration
SERIAL_PORT = 'COM4'  # Replace with your serial port
BAUD_RATE = 9600

# Initialize oscilloscope
rm = pyvisa.ResourceManager()
resources = rm.list_resources()
if not resources:
    raise Exception("No VISA resources found. Make sure the oscilloscope is connected.")
scope = rm.open_resource(resources[0])
scope.timeout = 5000

# Initial oscilloscope setup
channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
for channel in channels:
    scope.write(f':{channel}:DISPlay ON')
    scope.write(f':{channel}:SCALe 2')
    scope.write(f':{channel}:OFFSet -6')

# Heater values for all 40 heaters
heater_values = {i: 0.0 for i in range(40)}

# Fixed first layer (heaters 33-39)
fixed_first_layer_heaters = list(range(33, 40))
for heater in fixed_first_layer_heaters:
    heater_values[heater] = 0.01

# Input heaters (36 and 37)
input_heaters = [36, 37]
input_combinations = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]

# Modifiable heaters (all except 33-39)
modifiable_heaters = [i for i in range(40) if i not in fixed_first_layer_heaters]

# Voltage range for modifiable heaters
voltage_range = [0.1, 1.0, 2.0, 3.0, 4.0, 4.9]

# Global serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

def send_heater_values(ser, config):
    """Send heater values via serial connection"""
    voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.1)

def measure_outputs():
    """Measure outputs from the oscilloscope"""
    try:
        output1 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel1')), 5)
        output2 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel2')), 5)
        output3 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel3')), 5)
        output4 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel4')), 5)
        return [output1, output2, output3, output4]
    except Exception as e:
        print(f"Error measuring outputs: {e}")
        return [None, None, None, None]

def objective_function(heater_config):
    """
    Objective function for optimization.
    Evaluates the decoder-like behavior of a given heater configuration.
    
    Args:
    heater_config (list): Configuration of heater voltages
    
    Returns:
    float: Score representing how well the configuration behaves like a decoder
    """
    # Create a configuration dictionary
    config = {str(heater): float(value) for heater, value in zip(modifiable_heaters, heater_config)}
    
    # Add fixed heaters
    for heater in fixed_first_layer_heaters:
        config[str(heater)] = 0.01
    
    # Store results for each input combination
    iteration_results = []
    
    # Iterate through input combinations
    for input_state in input_combinations:
        # Create a copy of the configuration
        current_config = config.copy()
        
        # Set input heaters
        current_config['36'] = input_state[0]
        current_config['37'] = input_state[1]
        
        # Send heater values
        send_heater_values(ser, current_config)
        time.sleep(2)  # Allow time for stabilization
        
        # Measure outputs
        outputs = measure_outputs()
        
        if outputs == [None, None, None, None]:
            # If measurement fails, return a very low score
            return -1000
        
        # Store results
        iteration_results.append({
            'inputs': input_state,
            'outputs': outputs
        })
    
    # Score calculation
    score = evaluate_decoder_behavior(iteration_results)
    
    return score

def evaluate_decoder_behavior(results):
    """
    Evaluate how well the results resemble a decoder's behavior
    
    Args:
    results (list): List of dictionaries with input states and outputs
    
    Returns:
    float: Score representing decoder-like behavior
    """
    # Initialize score
    total_score = 0
    
    # Penalty threshold
    THRESHOLD = 1.0  # Voltage difference to consider significant
    
    for result in results:
        outputs = result['outputs']
        input_state = result['inputs']
        
        # Find the maximum output
        max_output = max(outputs)
        max_index = outputs.index(max_output)
        
        # Check if the max output is distinctly higher than others
        other_outputs = outputs.copy()
        other_outputs.pop(max_index)
        
        # Check if max output is significantly higher than others
        if all(max_output - out >= THRESHOLD for out in other_outputs):
            # Additional scoring based on input state
            # Prefer unique dominant outputs for different input states
            total_score += 10
        else:
            # Penalize configurations that don't show clear dominance
            total_score -= 5
    
    return total_score

def systematic_optimization():
    """
    Perform systematic optimization using differential evolution
    with reduced runtime and a callback function to track progress.
    """
    # Define bounds for modifiable heaters (0.1V to 4.9V)
    bounds = [(0.1, 4.9)] * len(modifiable_heaters)
    
    # Perform differential evolution
    result = differential_evolution(
        lambda x: -objective_function(x),  # Negative because scipy minimizes
        bounds,
        strategy='best1bin',
        maxiter=50,      # Reduced from 100 to speed up
        popsize=10,      # Reduced from 15 to speed up
        tol=0.05,        # Increased tolerance for faster convergence
        recombination=0.7,
        mutation=0.8,
        callback=optimization_progress,  # Track progress
    )
    
    # Return the best configuration and score
    best_config = {str(heater): value for heater, value in zip(modifiable_heaters, result.x)}
    return best_config, -result.fun

progress = []

def optimization_progress(xk, convergence):
    """Callback function to monitor optimization progress."""
    iteration_number = len(progress) + 1
    current_score = -objective_function(xk)  # Get the current best score
    print(f"Iteration {iteration_number}: Best Score = {current_score:.5f}, Convergence = {convergence:.5f}")
    progress.append((iteration_number, current_score, convergence))


def main():
    try:
        print("Starting systematic optimization...")
        
        # Perform optimization
        best_configuration, best_score = systematic_optimization()
        
        print("\nOptimization Complete!")
        print("Best Configuration Found:")
        for heater, value in best_configuration.items():
            print(f"Heater {heater}: {value}")
        
        print(f"\nObjective Function Score: {-best_score}")
        
        # Save best configuration
        with open("best_systematic_configuration.json", 'w') as f:
            json.dump(best_configuration, f, indent=4)
        
        # Detailed testing of the best configuration
        print("\nDetailed Testing of Best Configuration:")
        for input_state in input_combinations:
            # Set input heaters
            current_config = best_configuration.copy()
            current_config['36'] = input_state[0]
            current_config['37'] = input_state[1]
            
            # Send heater values
            send_heater_values(ser, current_config)
            time.sleep(2)  # Allow time for stabilization
            
            # Measure outputs
            outputs = measure_outputs()
            
            print(f"\nInput State (A, B): {input_state}")
            print(f"Outputs: O1={outputs[0]}, O2={outputs[1]}, O3={outputs[2]}, O4={outputs[3]}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Close connections
        ser.close()
        scope.close()

if __name__ == "__main__":
    main()