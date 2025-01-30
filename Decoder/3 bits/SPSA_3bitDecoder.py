import numpy as np
import serial
import time
import pyvisa
import random

# Serial port configuration 
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Core configuration
INPUT_HEATERS = [35, 36, 37]  # Three input heaters for 3-bit decoder
INPUT_STATES = [
    (0.1, 0.1, 0.1),  # 000
    (0.1, 0.1, 4.9),  # 001
    (0.1, 4.9, 0.1),  # 010
    (0.1, 4.9, 4.9),  # 011
    (4.9, 0.1, 0.1),  # 100
    (4.9, 0.1, 4.9),  # 101
    (4.9, 4.9, 0.1)   # 110
    # Note: 111 is not used as we only have 7 outputs
]

class DecoderOptimizer:
    def __init__(self):
        self.scopes = self._init_scopes()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
    def _init_scopes(self):
        rm = pyvisa.ResourceManager()
        SCOPE1_ID = 'USB0::0x1AB1::0x0610::HDO1B244000779::INSTR'
        SCOPE2_ID = 'USB0::0x1AB1::0x0610::HDO1B244100809::INSTR'
        
        scope1 = rm.open_resource(SCOPE1_ID)
        scope2 = rm.open_resource(SCOPE2_ID)
        scopes = [scope1, scope2]
        
        for scope in scopes:
            scope.timeout = 5000
            for i in range(1, 5):
                scope.write(f':CHANnel{i}:DISPlay ON')
                scope.write(f':CHANnel{i}:SCALe 2')
                scope.write(f':CHANnel{i}:OFFSet -6')
        
        return scopes

    def measure_outputs(self):
        """Measure outputs from both oscilloscopes"""
        try:
            outputs = []
            # First scope - all 4 channels
            for channel in range(1, 5):
                value = float(self.scopes[0].query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                outputs.append(round(value, 5))
            # Second scope - first 3 channels only
            for channel in range(1, 4):
                value = float(self.scopes[1].query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                outputs.append(round(value, 5))
            return outputs
        except Exception as e:
            print(f"Measurement error: {e}")
            return [None] * 7

    def send_heater_values(self, config):
        """Send heater values via serial connection"""
        message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
        self.serial.write(message.encode())
        self.serial.flush()
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        time.sleep(0.01)

    def evaluate_config(self, config, input_state):
        """Evaluate single configuration with given input state"""
        test_config = config.copy()
        test_config[INPUT_HEATERS[0]] = input_state[0]
        test_config[INPUT_HEATERS[1]] = input_state[1]
        test_config[INPUT_HEATERS[2]] = input_state[2]
        
        self.send_heater_values(test_config)
        time.sleep(0.25)
        raw_outputs = self.measure_outputs()
        
        if None in raw_outputs:
            return None
        
        # Convert outputs to probabilities for normalization
        outputs_array = np.array(raw_outputs)
        probs = outputs_array / np.sum(outputs_array)
        
        return raw_outputs, probs

    def calculate_loss(self, raw_outputs, target_idx):
        """Calculate loss based on separation between target and other outputs"""
        if raw_outputs is None:
            return float('inf')
            
        target_voltage = raw_outputs[target_idx]
        other_voltages = [v for i, v in enumerate(raw_outputs) if i != target_idx]
        max_other_voltage = max(other_voltages)
        
        # Calculate ratio (smaller is better)
        ratio = max_other_voltage / (target_voltage + 1e-10)
        return ratio

    def calculate_batch_loss(self, config):
        """Calculate total loss across all input states"""
        total_loss = 0
        expected_outputs = {
            (0.1, 0.1, 0.1): 0,  # 000
            (0.1, 0.1, 4.9): 1,  # 001
            (0.1, 4.9, 0.1): 2,  # 010
            (0.1, 4.9, 4.9): 3,  # 011
            (4.9, 0.1, 0.1): 4,  # 100
            (4.9, 0.1, 4.9): 5,  # 101
            (4.9, 4.9, 0.1): 6   # 110
        }
        
        for input_state in INPUT_STATES:
            raw_outputs, probs = self.evaluate_config(config, input_state)
            if raw_outputs is None:
                return float('inf')
                
            target_idx = expected_outputs[input_state]
            total_loss += self.calculate_loss(raw_outputs, target_idx)
            
        return total_loss / len(INPUT_STATES)

    def optimize(self, iterations=100, initial_delta=0.2, initial_learning_rate=0.05, decay_rate=50):
        """SPSA optimization for 3-bit decoder"""
        # Initialize weights for all modifiable heaters
        modifiable_heaters = [i for i in range(40) if i not in INPUT_HEATERS]
        w = {h: random.uniform(0.1, 4.9) for h in modifiable_heaters}
        
        # Fixed first layer values
        fixed_first_layer = list(range(33, 40))
        for heater in fixed_first_layer:
            if heater not in INPUT_HEATERS:
                w[heater] = 0.01
        
        delta = initial_delta
        learning_rate = initial_learning_rate
        best_config = w.copy()
        best_loss = float('inf')
        plateau_counter = 0
        current_best = float('inf')
        
        print("Starting SPSA optimization...")
        
        for iteration in range(iterations):
            # Decay parameters
            if delta > 0.11:
                delta /= (1 + iteration / decay_rate)
                learning_rate /= (1 + iteration / decay_rate)

            # Generate perturbation vector
            delta_vector = {h: random.choice([-1, 1]) for h in modifiable_heaters}
            
            # Create perturbed configurations
            w_plus = {h: min(max(v + delta * delta_vector[h], 0.1), 4.9) 
                     for h, v in w.items()}
            w_minus = {h: min(max(v - delta * delta_vector[h], 0.1), 4.9)
                      for h, v in w.items()}
            
            # Evaluate both configurations
            L_plus = self.calculate_batch_loss(w_plus)
            L_minus = self.calculate_batch_loss(w_minus)
            
            # Update configuration
            for h in modifiable_heaters:
                if h not in fixed_first_layer:  # Don't update fixed layer
                    gradient = (L_plus - L_minus) / (2 * delta * delta_vector[h])
                    w[h] = min(max(w[h] - learning_rate * gradient, 0.1), 4.9)

            # Track best configuration
            loss_now = min(L_minus, L_plus)
            previous_best = current_best
            current_best = min(best_loss, loss_now)
            
            # Handle plateaus
            if delta < 0.11:
                if abs(previous_best - current_best) < 1e-6:
                    plateau_counter += 1
                    if plateau_counter > 5:
                        delta += 0.5
                        learning_rate += 0.05
                        plateau_counter = 0
                else:
                    plateau_counter = 0
            
            print(f"Iteration {iteration + 1}: Best loss = {current_best:.4f} "
                  f"Current loss = {loss_now:.4f} Delta = {delta:.3f}, "
                  f"Learning Rate = {learning_rate:.3f}")
            
            if L_plus < best_loss:
                best_loss = L_plus
                best_config = w_plus.copy()
            if L_minus < best_loss:
                best_loss = L_minus
                best_config = w_minus.copy()
                
        return best_config, best_loss

    def cleanup(self):
        """Clean up resources"""
        self.serial.close()
        for scope in self.scopes:
            scope.close()

    def format_config(self, config):
        """Format configuration values"""
        return {k: round(float(v), 2) for k, v in config.items()}

def main():
    optimizer = DecoderOptimizer()
    try:
        best_config, best_loss = optimizer.optimize(
            iterations=100,
            initial_delta=0.7,
            initial_learning_rate=0.3,
            decay_rate=500
        )
        
        print(f"\nOptimization complete! Final loss: {best_loss:.4f}")
        clean_config = optimizer.format_config(best_config)
        print("\nFinal Heater Configuration:")
        for heater in sorted(clean_config.keys()):
            print(f"Heater {heater}: {clean_config[heater]:.2f}V")

        print("\nTesting final configuration:")
        for input_state in INPUT_STATES:
            current_config = clean_config.copy()
            for i, value in enumerate(input_state):
                current_config[INPUT_HEATERS[i]] = value
            
            optimizer.send_heater_values(current_config)
            time.sleep(0.25)
            outputs = optimizer.measure_outputs()
            
            max_output = max(outputs)
            max_index = outputs.index(max_output)
            
            print(f"\nInputs (A, B, C): {input_state}")
            print(f"Outputs: {outputs}")
            print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
            other_outputs = outputs.copy()
            other_outputs.pop(max_index)
            print(f"Separation from next highest: {(max_output - max(other_outputs)):.4f}V")
        
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    main()