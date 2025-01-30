from scipy.optimize import minimize
import random
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from typing import Dict, Tuple, List
import json
import time
from typing import Dict, List, Tuple, Optional
import pyvisa
import serial


# Constants
VOLTAGE_MIN = 0.1
VOLTAGE_MAX = 4.9
NUM_HEATERS = 40
GENERATION_LIMIT = 25
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
CACHE_SIZE = 1024
INPUT_COMBINATIONS = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]

class ConfigurationManager:
    """Manages configuration generation and validation"""
    def __init__(self):
        self.modifiable_heaters = sorted([i for i in range(33) if i not in [36, 37]], reverse=True)
        self.fixed_first_layer = list(range(33, 40))
    
    def generate_random_config(self) -> Dict[str, float]:
        """Generate a random configuration with continuous voltage values."""
        config = {
            str(h): round(random.uniform(VOLTAGE_MIN, VOLTAGE_MAX), 3) 
            for h in self.modifiable_heaters
        }
        config.update({
            str(h): round(random.uniform(0.01, 0.5), 3)
            for h in self.fixed_first_layer
        })
        return config


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

    def evaluate_single_input(self, config_str: str, input_state: tuple) -> float:
        config = json.loads(config_str)
        config["36"] = input_state[0]
        config["37"] = input_state[1]
        
        self.send_heater_values(config)
        outputs = self.measure_outputs()
        
        if None in outputs:
            return -10.0
        
        # Get target channel and highest output
        target_idx = {
            (0.1, 0.1): 0,
            (0.1, 4.9): 1,
            (4.9, 0.1): 2,  # Channel 3 for high-low
            (4.9, 4.9): 3   # Channel 4 for high-high
        }[input_state]
        
        max_output = max(outputs)
        actual_highest = outputs.index(max_output)
        
        # If wrong channel is highest, return negative score
        if actual_highest != target_idx:
            return -20.0  # Penalty for wrong channel
            
        # Base score for correct channel
        score = 10
        
        # Bonus points for separation ONLY if correct channel
        other_outputs = outputs.copy()
        other_outputs.pop(actual_highest)
        separation = max_output - max(other_outputs)
        bonus = min(separation * 5, 10)  # Cap bonus at 10 points
        return score + bonus

    
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

class HeaterLearner:
    def __init__(self, hardware, config_manager, num_heaters=40, voltage_steps=10):
        self.hardware = hardware
        self.config_manager = config_manager
        self.num_heaters = num_heaters
        self.voltage_steps = voltage_steps
        self.learning_rate = 0.5
        self.discount_factor = 0.9
        
        # Initialize Q-table
        # Key: (heater_index, current_voltage_step, input_state)
        # Value: dictionary of action -> expected_reward
        self.q_table = {}
        
        # Discretize voltage range into steps
        self.voltage_range = np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, voltage_steps)
    
    def get_voltage_step(self, voltage):
        return np.argmin(np.abs(self.voltage_range - voltage))
    
    def get_state_key(self, config, input_state):
        # Convert current configuration to discrete state
        voltage_steps = tuple(self.get_voltage_step(float(config[str(h)])) 
                            for h in range(self.num_heaters))
        return (voltage_steps, input_state)
    
    def get_actions(self, heater_index, current_step):
        actions = []
        if current_step > 0:
            actions.append('decrease')
        if current_step < self.voltage_steps - 1:
            actions.append('increase')
        return actions
    
    def learn(self, num_episodes=1000):
        best_config = None
        best_score = float('-inf')
        
        for episode in range(num_episodes):
            # Start with random configuration
            current_config = self.config_manager.generate_random_config()
            
            # Try each input condition
            for input_state in INPUT_COMBINATIONS:
                # Update config with input state
                print(input_state)
                config_copy = current_config.copy()
                config_copy["36"] = input_state[0]
                config_copy["37"] = input_state[1]
                
                # Choose and take action
                for heater in range(33):  # Exclude input heaters
                    state_key = self.get_state_key(config_copy, input_state)
                    current_step = self.get_voltage_step(float(config_copy[str(heater)]))
                    
                    # Get possible actions
                    actions = self.get_actions(heater, current_step)
                    
                    # Initialize Q-values if not seen before
                    if state_key not in self.q_table:
                        self.q_table[state_key] = {action: 0.0 for action in actions}
                    
                    # Choose action (epsilon-greedy)
                    if random.random() < 0.3:  # exploration
                        action = random.choice(actions)
                    else:  # exploitation
                        action = max(actions, key=lambda a: self.q_table[state_key].get(a, 0.0))
                    
                    # Take action
                    new_config = config_copy.copy()
                    if action == 'increase':
                        new_step = min(current_step + 1, self.voltage_steps - 1)
                    else:
                        new_step = max(current_step - 1, 0)
                    new_config[str(heater)] = self.voltage_range[new_step]
                    
                    # Get reward
                    reward = self.hardware.evaluate_single_input(
                        json.dumps(new_config), input_state
                    )
                    print(reward)
                    # Update Q-value
                    old_value = self.q_table[state_key].get(action, 0.0)
                    next_state_key = self.get_state_key(new_config, input_state)
                    next_max = max(self.q_table.get(next_state_key, {}).values(), default=0)
                    
                    new_value = (1 - self.learning_rate) * old_value + \
                               self.learning_rate * (reward + self.discount_factor * next_max)
                    
                    self.q_table[state_key][action] = new_value
                    
                    # Update configuration if better
                    if reward > best_score:
                        best_score = reward
                        best_config = new_config.copy()
                        print(f"Episode {episode}, New best score: {best_score}")
                    
                    current_config = new_config
        
        return best_config, best_score
    

def main():

    print("Starting  Optimization...")
    config_manager = ConfigurationManager()
    hardware = HardwareInterface()
    optimizer = HeaterLearner(hardware, config_manager)

    best_config, best_score = optimizer.learn()

    print("\nOptimization Complete!")
    print(f"Best Score: {best_score}")
    print(f"Best Configuration: {best_config}")



if __name__ == "__main__":
    main()