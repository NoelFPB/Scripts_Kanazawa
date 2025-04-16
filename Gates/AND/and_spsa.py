# The results of running this script are not good, the AND gate never works properly
# Different loss functions have been tried, no good result



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
MODIFIABLE_HEATERS = [i for i in range(40) if i not in INPUT_HEATERS]
FIXED_FIRST_LAYER = list(range(33, 40))

# Expected outputs for AND gate  only checking one output channel
# For AND gate, output should be high (1) only when both inputs are high (4.9, 4.9)
EXPECTED_OUTPUTS = [0, 0, 0, 1]  # 0=low, 1=high

class AndGateOptimizer:
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
        
        # Only set up channel 1 for AND gate output
        scope.write(':CHANnel1:DISPlay ON')
        scope.write(':CHANnel1:SCALe 2')
        scope.write(':CHANnel1:OFFSet -6')
        
        # Turn off other channels as we don't need them
        for i in range(2, 5):
            scope.write(f':CHANnel{i}:DISPlay OFF')
            
        return scope

    def measure_output(self):
        """Measure just the AND gate output from oscilloscope (Channel 1)"""
        try:
            value = float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel1'))
            return round(value, 5)
        except Exception as e:
            print(f"Measurement error: {e}")
            return None

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
        time.sleep(0.2)
        output = self.measure_output()
        
        if output is None:
            return float('inf')  # Return a large loss value on error
            
        return output

    def calculate_loss(self, output, expected):
            """
            Calculate loss for AND gate:
            - If expected is 1 (HIGH): loss is lower when output is higher (above threshold)
            - If expected is 0 (LOW): loss is lower when output is lower (below threshold)
            """
            HIGH_THRESHOLD = 4
            LOW_THRESHOLD = 1.0

            if expected == 1:  # Should be HIGH
                if output > HIGH_THRESHOLD:
                    # Reward higher voltage with lower loss
                    return 1.0 / (output + 0.1)  # Add 0.1 to avoid division by zero
                else:
                    # Penalize if not high enough
                    return 10.0 + (HIGH_THRESHOLD - output)
            else:  # Should be LOW
                if output < LOW_THRESHOLD:
                    # Reward lower voltage with lower loss
                    return output
                else:
                    # Penalize if not low enough
                    return 10.0 + (output - LOW_THRESHOLD)     



    def calculate_batch_loss(self, config):
        """Calculate total loss across all input states"""
        total_loss = 0
        results = []
        
        for i, input_state in enumerate(INPUT_STATES):
            # Evaluate the configuration for the current input state
            output = self.evaluate_config(config, input_state)
            expected = EXPECTED_OUTPUTS[i]
            
            # Calculate the loss for the current state
            loss = self.calculate_loss(output, expected)
            total_loss += loss
            
            # Store results for later analysis
            results.append({
                'inputs': input_state,
                'output': output,
                'expected': 'HIGH' if expected == 1 else 'LOW',
                'actual': 'HIGH' if output > 4.0 else 'LOW',
                'loss': loss
            })
            
        # Return the average loss across all input states
        return total_loss / len(INPUT_STATES), results
    
    def optimize(self, iterations=80, initial_delta=0.8, initial_learning_rate=0.2, step_size=0.1):
        """
        SPSA optimization for AND gate with simple decaying hyperparameters
        and discretization by step size
        """
        # Helper function to discretize a value to the nearest step
        def discretize(value, step=step_size):
            return round((value / step) * step,2)
        
        # Initialize with random values, discretized to the step size
        w = {h: discretize(random.uniform(0.0, 4.9)) for h in MODIFIABLE_HEATERS}
        print(w)
        
        # Initialize optimization parameters
        delta = initial_delta
        learning_rate = initial_learning_rate
        best_config = w.copy()
        best_loss = float('inf')
        
        print("Starting discrete SPSA optimization for AND gate...")
        
        for iteration in range(iterations):
            # Generate random perturbation direction
            delta_vector = {h: random.choice([-1, 1]) for h in MODIFIABLE_HEATERS}
            # Create perturbed configurations
            w_plus = {h: min(max(v + delta * delta_vector[h], 0.0), 4.9) 
                    for h, v in w.items()}
            w_minus = {h: min(max(v - delta * delta_vector[h], 0.0), 4.9)
                    for h, v in w.items()}
            
            # Discretize the perturbed configurations
            w_plus = {h: discretize(v) for h, v in w_plus.items()}
            w_minus = {h: discretize(v) for h, v in w_minus.items()}
            
          # Evaluate perturbed configurations
            L_plus, _ = self.calculate_batch_loss(w_plus)
            L_minus, _ = self.calculate_batch_loss(w_minus)
                    
            # Update configuration using gradient approximation
            for h in MODIFIABLE_HEATERS:
                gradient = (L_plus - L_minus) / (2 * delta * delta_vector[h])
                w[h] -= learning_rate * gradient
                w[h] = min(max(w[h], 0.0), 4.9)  # Clamp values to valid range
                
                # Discretize the updated value
                w[h] = discretize(w[h])
            
            # Simple parameter decay
            delta *= 0.99
            learning_rate *= 0.99
            
            # Update best configuration if improved
            current_loss = min(L_plus, L_minus)
            if current_loss < best_loss:
                best_loss = current_loss
                best_config = w_plus.copy() if L_plus < L_minus else w_minus.copy()
            

            # Print progress
            print(f"Iteration {iteration + 1}: Best loss = {best_loss:.4f}, Current loss = {current_loss:.4f}, Delta = {delta:.3f}, Learning Rate = {learning_rate:.3f}")
        return best_config, best_loss

    def test_configuration(self, config):
        """Test and print performance of a configuration"""
        print("\nTesting AND gate configuration:")
        all_correct = True
        
        for i, input_state in enumerate(INPUT_STATES):
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            output = self.measure_output()
            
            expected = EXPECTED_OUTPUTS[i]
            actual_state = "HIGH" if output > 2.0 else "LOW"
            expected_state = "HIGH" if expected == 1 else "LOW"
            is_correct = (actual_state == expected_state)
            
            if not is_correct:
                all_correct = False
                
            print(f"\nInputs (A, B): {input_state}")
            print(f"Output: {output:.4f}V")
            print(f"Output is: {actual_state}")
            print(f"Expected: {expected_state}")
            print(f"Correct: {'Yes' if is_correct else 'No'}")
            
        return all_correct

    def cleanup(self):
        """Close connections"""
        self.serial.close()
        self.scope.close()

    def format_config(self, config):
            """Format configuration values for display and ensure all heaters 0-39 are included"""
            # Start with a complete range of heaters
            complete_config = {i: 0.0 for i in range(40)}
            
            # Add all values from the optimization config
            for heater, value in config.items():
                complete_config[heater] = value
                
            # Make sure fixed first layer heaters are set correctly
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:  # Don't override input heaters
                    complete_config[h] = 0.01
                    
            # Format all values to 2 decimal places
            return {k: round(float(v), 2) for k, v in complete_config.items()}

def main():
    optimizer = AndGateOptimizer()
    try:
        # Run SPSA optimization
        best_config, best_loss = optimizer.optimize(iterations=100, 
                                                   initial_delta=0.5, 
                                                   initial_learning_rate=0.2)
        
        print(f"\nOptimization complete! Final loss: {best_loss:.4f}")
        
        # Clean up and display the best configuration
        clean_config = optimizer.format_config(best_config)
        print(clean_config)
     
        # Test the final configuration
        all_correct = optimizer.test_configuration(clean_config)
        
        if all_correct:
            print("\nSuccess! The AND gate is working correctly for all input combinations.")
        else:
            print("\nThe AND gate is not working perfectly for all input combinations.")
            print("You may want to run the optimization again or try different parameters.")
        
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    main()