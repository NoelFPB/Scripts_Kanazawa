import numpy as np
import serial
import time
import pyvisa
import random

# Couldnt get good results with this.

# Core configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
INPUT_HEATERS = [36, 37]
INPUT_STATES = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
MODIFIABLE_HEATERS = [i for i in range(40)]

class DecoderOptimizer:
    def __init__(self):
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
    def _init_scope(self):
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No oscilloscope found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
        
        for i in range(1, 5):
            scope.write(f':CHANnel{i}:DISPlay ON')
            scope.write(f':CHANnel{i}:SCALe 2')
            scope.write(f':CHANnel{i}:OFFSet -6')
        return scope

    def measure_outputs(self):
        outputs = []
        for channel in range(1, 5):
            value = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
            outputs.append(round(value, 5))
        return outputs

    def send_heater_values(self, config):
        message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
        self.serial.write(message.encode())
        self.serial.flush()
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        time.sleep(0.1)

    def evaluate_config(self, config, input_state):
        """Evaluate single configuration with given input state"""
        test_config = config.copy()
        test_config[INPUT_HEATERS[0]] = input_state[0]
        test_config[INPUT_HEATERS[1]] = input_state[1]
        
        self.send_heater_values(test_config)
        time.sleep(0.25)
        raw_outputs = self.measure_outputs()
        
        # Convert outputs to probabilities
        outputs_array = np.array(raw_outputs)
        probs = outputs_array / np.sum(outputs_array)
        
        return probs

    # def calculate_loss(self, probs, target_idx):
    #     """Calculate cross-entropy loss for single sample"""
    #     target = np.zeros(4)
    #     target[target_idx] = 1
    #     return -np.sum(target * np.log(probs + 1e-10))

    def calculate_loss(self, probs, target_idx):
        """Calculate modified cross-entropy loss with additional penalties"""
        target = np.zeros(4)
        target[target_idx] = 1
        
        # Basic cross-entropy
        ce_loss = -np.sum(target * np.log(probs + 1e-10))
        
        # Add penalty for incorrect outputs being too high
        incorrect_penalty = np.sum((1 - target) * probs)
        
        return ce_loss + 10.0 * incorrect_penalty

    def optimize(self, iterations=1000, delta=0.5, learning_rate=0.2):
        w = {h: random.uniform(0.1, 4.9) for h in MODIFIABLE_HEATERS}
        best_config = w.copy()
        best_loss = float('inf')
        
        for iteration in range(iterations):
            # Test each input state in sequence
            input_idx = iteration % len(INPUT_STATES)
            input_state = INPUT_STATES[input_idx]
            
            # Generate perturbation vector
            delta_vector = {h: random.choice([-1, 1]) for h in MODIFIABLE_HEATERS}
            
            # Create perturbed configurations
            w_plus = {h: min(max(v + delta * delta_vector[h], 0.1), 4.9) 
                     for h, v in w.items()}
            w_minus = {h: min(max(v - delta * delta_vector[h], 0.1), 4.9)
                      for h, v in w.items()}
            
            # Evaluate both configurations
            probs_plus = self.evaluate_config(w_plus, input_state)
            time.sleep(0.05) #This delay is not really necesarry
            probs_minus = self.evaluate_config(w_minus, input_state)
            
            # Calculate losses
            L_plus = self.calculate_loss(probs_plus, input_idx)
            L_minus = self.calculate_loss(probs_minus, input_idx)
            
            # Update configuration using gradient approximation
            for h in MODIFIABLE_HEATERS:
                gradient = (L_plus - L_minus) / (2 * delta * delta_vector[h])
                w[h] -= learning_rate * gradient  # Gradient descent
                w[h] = min(max(w[h], 0.1), 4.9)  # Clip values
            
            # Evaluate current configuration
            probs = self.evaluate_config(w, input_state)
            current_loss = self.calculate_loss(probs, input_idx)
            
            # Update best configuration if better
            if current_loss < best_loss:
                best_loss = current_loss
                best_config = w.copy()
            print(f"Iteration {iteration + 1}: Loss {current_loss:.4f}")
        
        return best_config, best_loss

    def cleanup(self):
        self.serial.close()
        self.scope.close()

    def format_config(self, config):
        return {k: round(float(v), 2) for k, v in config.items()}

def main():
    optimizer = DecoderOptimizer()
    try:
        print("Starting SPSA optimization...")
        best_config, best_loss = optimizer.optimize()
        print(f"\nOptimization complete! Final loss: {best_loss:.4f}")
        clean_config = optimizer.format_config(best_config)
        print(clean_config)

        print("\nTesting final configuration:")
        for input_state in INPUT_STATES:
            current_config = clean_config.copy()
            current_config[36] = input_state[0]
            current_config[37] = input_state[1]
            
            optimizer.send_heater_values(current_config)
            time.sleep(0.5)
            outputs = optimizer.measure_outputs()
            
            max_output = max(outputs)
            max_index = outputs.index(max_output)
            
            print(f"\nInputs (A, B): {input_state}")
            print(f"Outputs: {outputs}")
            print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
        
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    main()