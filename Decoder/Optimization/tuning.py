import serial
import time
import pyvisa
import json
import random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import numpy as np

# Enhanced Constants
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
CACHE_SIZE = 1024
# Create voltage options from 0.1 to 4.9 with 0.1 step
VOLTAGE_OPTIONS = [round(v/10, 1) for v in range(1, 50)]
# This gives us: [0.1, 0.2, 0.3, ..., 4.8, 4.9]
INPUT_HEATERS = [36, 37]
EXPECTED_OUTPUTS = [
    [1, 0, 0, 0],  # For input (0.1, 0.1) -> strongest on channel 1
    [0, 1, 0, 0],  # For input (0.1, 4.9) -> strongest on channel 2
    [0, 0, 1, 0],  # For input (4.9, 0.1) -> strongest on channel 3
    [0, 0, 0, 1],  # For input (4.9, 4.9) -> strongest on channel 4
]

INPUT_COMBINATIONS = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
INPUT_STATES = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
MODIFIABLE_HEATERS = [i for i in range(40)]

class HardwareInterface:
    """Manages hardware communication"""
    def __init__(self):
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        self.scope = rm.open_resource(resources[0])
        self.scope.timeout = 5000
        
        self._setup_channels()
    
    def _setup_channels(self):
        """Configure oscilloscope channels"""
        for channel in ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']:
            self.scope.write(f':{channel}:DISPlay ON')
            self.scope.write(f':{channel}:SCALe 2')
            self.scope.write(f':{channel}:OFFSet -6')
    
    def send_heater_values(self, config: Dict[str, float]) -> None:
        """Send configuration to hardware"""
        voltage_message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
        self.ser.write(voltage_message.encode())
        self.ser.flush()
        time.sleep(0.25)  # Reduced delay 
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
    
    def measure_outputs(self) -> List[Optional[float]]:
        """Measure all channels with error handling"""
        try:
            outputs = []
            for channel in range(1, 5):
                value = float(self.scope.query(
                    f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'
                ))
                outputs.append(round(value, 5))
            return outputs
        except Exception as e:
            print(f"Measurement error: {e}")
            return [None] * 4
    
    def cleanup(self):
        """Clean up hardware connections"""
        self.ser.close()
        self.scope.close()




    def evaluate_config(self, config, input_state):
        """Evaluate single configuration with given input state"""
        test_config = config.copy()
        test_config[INPUT_HEATERS[0]] = input_state[0]
        test_config[INPUT_HEATERS[1]] = input_state[1]
        
        self.send_heater_values(test_config)
        time.sleep(0.25)
        raw_outputs = self.measure_outputs()
        print(raw_outputs)
        
        # Convert outputs to probabilities
        outputs_array = np.array(raw_outputs)
        probs = outputs_array / np.sum(outputs_array)
        
        return probs

def tune_configuration(initial_config: Dict[str, float], hardware: HardwareInterface, 
                      num_iterations: int = 100, voltage_step: float = 0.1):
    """
    Fine-tune a known good configuration by making small adjustments.
    """
    best_config = initial_config.copy()
    best_score = evaluate_configuration(best_config)
    print(f"Initial Score: {best_score}")
    
    # Try tuning each heater
    for iteration in range(num_iterations):
        # Pick a random heater to tune
        heater = str(random.randint(0, 39))
        original_value = best_config[heater]
        
        # Try slightly higher and lower voltages
        for new_value in [
            round(original_value + voltage_step, 1),
            round(original_value - voltage_step, 1)
        ]:
            # Check voltage is in valid range
            if 0.1 <= new_value <= 4.9:
                test_config = best_config.copy()
                test_config[heater] = new_value
                
                score = evaluate_configuration(test_config)
                
                if score > best_score:
                    print(f"Improvement found! Heater {heater}: {original_value}V -> {new_value}V")
                    print(f"New score: {score} (was {best_score})")
                    best_config = test_config
                    best_score = score
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Current best score: {best_score}")
    
    return best_config, best_score

def calculate_loss(probs, target_idx):
    """Calculate loss with heavy penalty for wrong channel"""
    # Get highest probability channel
    actual_highest = int(np.argmax(probs))  # Convert to int
    print(actual_highest)
    print(probs)
    # Base loss using cross-entropy
    target = np.zeros(4)
    target[target_idx] = 1
    cross_entropy = -np.sum(target * np.log(probs + 1e-10))
    
    # Add large penalty if wrong channel is highest
    if actual_highest != target_idx:
        penalty = 10.0  # Large penalty value
        return cross_entropy + penalty
    
    return cross_entropy

def tune_with_spsa(initial_config, iterations=100, delta=0.1, learning_rate=0.01):
    """
    Tune existing configuration using SPSA
    
    Args:
        initial_config: Your good configuration to start from
        iterations: Number of SPSA iterations
        delta: Size of perturbation (smaller for fine-tuning)
        learning_rate: Step size (smaller for fine-tuning)
    """
    w = initial_config.copy()
    best_config = w.copy()
    best_loss = float('inf')
    
    for iteration in range(iterations):
        # Generate perturbation vector using string keys
        delta_vector = {str(h): random.choice([-1, 1]) for h in range(40)}
        
        # Create perturbed configurations
        w_plus = {h: min(max(v + delta * delta_vector[h], 0.1), 4.9) 
                for h, v in w.items()}
        w_minus = {h: min(max(v - delta * delta_vector[h], 0.1), 4.9)
                for h, v in w.items()}
        
        # Rest of the function remains the same...
        
        L_plus = 0
        L_minus = 0
        
        # Evaluate both perturbations
        for input_state, target_dist in zip(INPUT_STATES, EXPECTED_OUTPUTS):
            probs_plus = hardware.evaluate_config(w_plus, input_state)
            L_plus += calculate_loss(probs_plus, target_dist)
            
            probs_minus = hardware.evaluate_config(w_minus, input_state)
            L_minus += calculate_loss(probs_minus, target_dist)

        L_plus /= len(INPUT_STATES)
        L_minus /= len(INPUT_STATES)
                
        # Update all parameters simultaneously
        for h in w.keys():
            gradient = (L_plus - L_minus) / (2 * delta * delta_vector[h])
            w[h] -= learning_rate * gradient
            w[h] = min(max(w[h], 0.1), 4.9)
         
        # Track best configuration
        current_best = min(best_loss, L_plus, L_minus)
        loss_now = min(L_minus, L_plus)
        print(f"Iteration {iteration + 1}: Best loss = {current_best:.4f} Current loss = {loss_now:.4f}")

        if L_plus < best_loss:
            best_loss = L_plus
            best_config = w_plus.copy()
        if L_minus < best_loss:
            best_loss = L_minus
            best_config = w_minus.copy()
            
    return best_config, best_loss

def evaluate_configuration(config: Dict[str, float]) -> float:
    """
    Evaluate a configuration (using your existing scoring method)
    """
    total_score = 0
    for input_state in INPUT_COMBINATIONS:
        test_config = config.copy()
        test_config["36"] = input_state[0]
        test_config["37"] = input_state[1]
        
        hardware.send_heater_values(test_config)
        time.sleep(0.25)
        outputs = hardware.measure_outputs()
        
        if None in outputs:
            return -10.0
        
        target_idx = {
            (0.1, 0.1): 0,
            (0.1, 4.9): 1,
            (4.9, 0.1): 2,
            (4.9, 4.9): 3
        }[input_state]
        
        max_output = max(outputs)
        actual_highest = outputs.index(max_output)
        
        if actual_highest == target_idx:
            score = 10
            other_outputs = outputs.copy()
            other_outputs.pop(actual_highest)
            separation = max_output - max(other_outputs)
            bonus = separation * 5
            total_score += score + bonus
            
    return total_score

# Usage example:
your_good_config = {'0': 3.6, '1': 2.0, '2': 3.2, '3': 0.8, '4': 3.6, '5': 2.6, '6': 1.3, '7': 3.5, '8': 0.4, '9': 1.2, '10': 1.1, '11': 2.5, '12': 0.9, '13': 3.5, '14': 3.7, '15': 1.8, '16': 2.8, '17': 3.0, '18': 3.4, '19': 1.8, '20': 3.9, '21': 4.0, '22': 4.1, '23': 2.6, '24': 2.5, '25': 4.0, '26': 3.6, '27': 3.4, '28': 3.8, '29': 1.0, '30': 2.7, '31': 2.1, '32': 3.7, '33': 0.1, '34': 0.1, '35': 0.01, '36': 0.1, '37': 0.1, '38': 0.01, '39': 0.1}

hardware = HardwareInterface()
        

# tuned_config, final_score = tune_configuration(
#     initial_config=your_good_config,
#     hardware=hardware,
#     num_iterations=100,  # Adjust as needed
#     voltage_step=0.1     # How much to adjust voltages by
# )


tuned_config, final_loss = tune_with_spsa(
    initial_config=your_good_config,
    iterations=50,      # Fewer iterations needed for tuning
    delta=0.1,         # Smaller perturbations for fine-tuning
    learning_rate=0.01  # Smaller steps for fine-tuning
)