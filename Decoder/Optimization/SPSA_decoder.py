import numpy as np
import serial
import time
import pyvisa
import random

# Core configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
INPUT_HEATERS = [36, 37]
INPUT_STATES = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
VOLTAGE_OPTIONS = [0.1, 2.5, 4.9]
MODIFIABLE_HEATERS = list(range(33))  # Only use heaters 0-32 for configuration

class DecoderOptimizer:
    def __init__(self):
        # Initialize hardware connections
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)  # Brief settling time
        
    def _init_scope(self):
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No oscilloscope found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
        
        # Configure all channels
        for i in range(1, 5):
            scope.write(f':CHANnel{i}:DISPlay ON')
            scope.write(f':CHANnel{i}:SCALe 2')
        return scope

    def measure_outputs(self):
        """Get voltage measurements from all 4 channels"""
        outputs = []
        for channel in range(1, 5):
            value = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
            outputs.append(round(value, 5))
        return outputs

    def send_heater_values(self, config):
        """Send heater configuration via serial"""
        message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
        self.serial.write(message.encode())
        self.serial.flush()
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(0.1)  # Reduced settling time

    def evaluate_config(self, config):
        """Calculate loss using cross-entropy after converting outputs to probabilities"""
        total_loss = 0
        
        for idx, input_state in enumerate(INPUT_STATES):
            # Apply input state
            test_config = config.copy()
            test_config[INPUT_HEATERS[0]] = input_state[0]
            test_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(test_config)
            time.sleep(0.25)
            raw_outputs = self.measure_outputs()
            
            # Convert outputs to probabilities using softmax
            outputs_array = np.array(raw_outputs)
            probs = outputs_array / np.sum(outputs_array)  # Normalize to probabilities
            
            # Create one-hot target vector
            target = np.zeros(4)
            target[idx] = 1
            
            # Calculate cross-entropy loss
            loss = -np.sum(target * np.log(probs + 1e-10))  # Add small epsilon to avoid log(0)
            total_loss += loss
        
        # Return negative loss since SPSA maximizes
        return -total_loss

    def optimize(self, iterations=30, delta=0.2, learning_rate=0.1):
        """Run SPSA optimization"""
        # Initialize random configuration
        best_config = {h: random.choice(VOLTAGE_OPTIONS) for h in MODIFIABLE_HEATERS}
        best_score = self.evaluate_config(best_config)
        
        for iteration in range(iterations):
            # Generate perturbation vector
            perturbations = {h: random.choice([-1, 1]) for h in MODIFIABLE_HEATERS}
            
            # Evaluate perturbed configurations
            config_plus = {h: min(max(v + delta * perturbations[h], 0.1), 4.9) 
                         for h, v in best_config.items()}
            config_minus = {h: min(max(v - delta * perturbations[h], 0.1), 4.9) 
                          for h, v in best_config.items()}
            
            score_plus = self.evaluate_config(config_plus)
            score_minus = self.evaluate_config(config_minus)
            
            # Update configuration based on gradient approximation
            for heater in MODIFIABLE_HEATERS:
                gradient = (score_plus - score_minus) / (2 * delta * perturbations[heater])
                current_value = best_config[heater]
                best_config[heater] = min(max(
                    current_value - learning_rate * gradient,
                    0.1), 4.9)
            
            # Evaluate new configuration
            current_score = self.evaluate_config(best_config)
            if current_score > best_score:  # Remember: score is negative loss
                best_score = current_score
                print(f"Iteration {iteration + 1}: Loss improved to {-best_score}")
          
        return best_config, best_score

    def cleanup(self):
        """Close connections"""
        self.serial.close()
        self.scope.close()

def main():
    optimizer = DecoderOptimizer()
    try:
        print("Starting SPSA optimization...")
        best_config, best_score = optimizer.optimize()
        print(f"\nOptimization complete! Final score: {best_score}")
        print(best_config)
        
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    main()