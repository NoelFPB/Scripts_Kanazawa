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

EXPECTED_OUTPUTS = [
    [1, 0, 0, 0],  # For input (0.1, 0.1) -> strongest on channel 1
    [0, 1, 0, 0],  # For input (0.1, 4.9) -> strongest on channel 2
    [0, 0, 1, 0],  # For input (4.9, 0.1) -> strongest on channel 3
    [0, 0, 0, 1],  # For input (4.9, 4.9) -> strongest on channel 4
]

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
    #     loss = -np.sum(target * np.log(probs + 1e-10))
    #     #print(loss)
    #     return loss
    
    def calculate_loss(self, raw_outputs, target_idx):
        """Loss based on ratio of target voltage to max other voltage"""
        target_voltage = raw_outputs[target_idx]
        other_voltages = [v for i, v in enumerate(raw_outputs) if i != target_idx]
        max_other_voltage = max(other_voltages)
        
        ratio = max_other_voltage / (target_voltage + 1e-10)  # Add small epsilon to avoid division by zero
        return ratio  # A smaller ratio means better separation
        
    def calculate_batch_loss(self, config):
        total_loss = 0
        for input_state, target_dist in zip(INPUT_STATES, EXPECTED_OUTPUTS):
            # Evaluate the configuration for the current input state
            probs = self.evaluate_config(config, input_state)
            # Calculate the loss for the current state
            total_loss += self.calculate_loss(probs, np.argmax(target_dist))
        # Return the average loss across all input states
        return total_loss / len(INPUT_STATES)
    
    def optimize(self, iterations=100, initial_delta=0.2, initial_learning_rate=0.05, decay_rate = 50):
        w = {h: random.uniform(0.1, 4.9) for h in MODIFIABLE_HEATERS}
        delta = initial_delta
        learning_rate = initial_learning_rate
        best_config = w.copy()
        best_loss = float('inf')
        a = 0
        current_best = 0
        
        for iteration in range(iterations):
            if delta > 0.11:
                delta /= (1 + iteration/ decay_rate)
                learning_rate /= (1 + iteration/ decay_rate)

            # Generate perturbation vector
            delta_vector = {h: random.choice([-1, 1]) for h in MODIFIABLE_HEATERS}
            
            # Create perturbed configurations
            w_plus = {h: min(max(v + delta * delta_vector[h], 0.1), 4.9) 
                    for h, v in w.items()}
            w_minus = {h: min(max(v - delta * delta_vector[h], 0.1), 4.9)
                    for h, v in w.items()}
            
            L_plus = 0
            L_minus = 0
            
            # Evaluate w_plus configuration
            L_plus += self.calculate_batch_loss(w_plus)
            
            # Evaluate w_minus configuration
            L_minus += self.calculate_batch_loss(w_minus)

            # Average the losses
            L_plus /= len(INPUT_STATES)
            L_minus /= len(INPUT_STATES)
                    
            # Update configuration
            for h in MODIFIABLE_HEATERS:
                gradient = (L_plus - L_minus) / (2 * delta * delta_vector[h])
                w[h] -= learning_rate * gradient
                w[h] = min(max(w[h], 0.1), 4.9)

            previous_best = current_best 
            current_best = min(best_loss, L_plus, L_minus)
            #print(previous_best)
            #print(current_best)
            if delta < 0.11:
                if previous_best == current_best:
                    a+=1
                    print(a)
                    if a > 5:
                        delta+=0.5
                        learning_rate+=0.05
                        a = 0
                else:
                    a = 0
            loss_now = min(L_minus, L_plus)
            print(f"Iteration {iteration + 1}: Best loss = {current_best:.4f} Current loss = {loss_now:.4f} Delta {delta:.3f}, Learnign Rate {learning_rate:.3f}")
            
            if L_plus < best_loss:
                best_loss = L_plus
                best_config = w_plus.copy()
            if L_minus < best_loss:
                best_loss = L_minus
                best_config = w_minus.copy()
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
        best_config, best_loss = optimizer.optimize(iterations=100, 
                                                    initial_delta=0.7, 
                                                    initial_learning_rate=0.3,
                                                    decay_rate=500)
        
        print(f"\nOptimization complete! Final loss: {best_loss:.4f}")
        clean_config = optimizer.format_config(best_config)
        print(clean_config)

        print("\nTesting final configuration:")
        for input_state in INPUT_STATES:
            current_config = clean_config.copy()
            current_config[36] = input_state[0]
            current_config[37] = input_state[1]
            
            optimizer.send_heater_values(current_config)
            time.sleep(0.25)
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