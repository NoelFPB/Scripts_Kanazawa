import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc  # For Latin Hypercube Sampling


# Serial port configuration 
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# === GATE CONFIGURATION ===
GATE_TYPE = "XNOR"  # Logic gate type to optimize (AND, OR, NAND, NOR, XOR, XNOR)
INPUT_HEATERS = [36, 37]  # Heaters for input A and B

# Voltage definitions
V_MIN = 0.1     # Representing logical LOW
V_MAX = 4.9    # Representing logical HIGH

# Heater configuration
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in INPUT_HEATERS]

# Input combinations for testing
INPUT_COMBINATIONS = [
    (V_MIN, V_MIN),
    (V_MIN, V_MAX),
    (V_MAX, V_MIN),
    (V_MAX, V_MAX)
]

# Generate truth table based on gate type
def generate_truth_table(gate_type):
    truth_table = {}
    if gate_type == "AND":
        outputs = [False, False, False, True]
    elif gate_type == "OR":
        outputs = [False, True, True, True]
    elif gate_type == "NAND":
        outputs = [True, True, True, False]
    elif gate_type == "NOR":
        outputs = [True, False, False, False]
    elif gate_type == "XOR":
        outputs = [False, True, True, False]
    elif gate_type == "XNOR":
        outputs = [True, False, False, True]
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")
    
    for i, input_pair in enumerate(INPUT_COMBINATIONS):
        truth_table[input_pair] = outputs[i]
    
    return truth_table

class SPSALogicGateOptimizer:
    """
    Optimize a physical logic gate using Simultaneous Perturbation Stochastic Approximation (SPSA).
    """
    def __init__(self, gate_type=GATE_TYPE):
        # Initialize gate type and truth table
        self.gate_type = gate_type
        self.truth_table = generate_truth_table(gate_type)
        
        print(f"Optimizing {gate_type} gate with truth table:")
        for inputs, output in self.truth_table.items():
            print(f"  {inputs} -> {'HIGH' if output else 'LOW'}")
        
        # Initialize hardware connections
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Best configuration found
        self.best_config = None
        self.best_score = float('-inf')

        self.base_config = {}
    
        # Set fixed first layer values once and forget about them
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                self.base_config[h] = 0.01

        
    def _init_scope(self):
        """Initialize oscilloscope for logic gate output measurement"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
        
        # Setup Channel 1 for logic gate output measurement
        scope.write(':CHANnel2:DISPlay ON')
        scope.write(':CHANnel2:SCALe 2')
        scope.write(':CHANnel2:OFFSet -6')
        
        # Turn off other channels
        # for channel_num in range(2, 5):
        #     scope.write(f':CHANnel{channel_num}:DISPlay OFF')
            
        return scope
    
    def send_heater_values(self, config):
        """Send heater voltage configuration to hardware"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.serial.write(voltage_message.encode())
        self.serial.flush()
        time.sleep(0.01)  # Small delay to ensure message is sent
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
    
    def measure_output(self):
        """Measure the logic gate output voltage from oscilloscope"""
        try:
            value = float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel2'))
            return round(value, 5)
        except Exception as e:
            print(f"Measurement error: {e}")
            return None
            
    def evaluate_configuration(self, config):
        """Evaluate configuration focusing purely on separation and consistency, independent of thresholds."""
        
        # Component weights (sum to 1.0) - Only focus on what matters for separation
        WEIGHTS = {
            'high_consistency': 0.2,  # Consistency of HIGH outputs
            'low_consistency': 0.1,   # Consistency of LOW outputs  
            'separation': 0.6,         # Primary: separation between HIGH/LOW
            'success_bonus': 0.1       # Bonus for correct relative ordering
        }
        
        # Initialize scores
        component_scores = {k: 0.0 for k in WEIGHTS}
        
        # Collect all outputs for analysis
        high_outputs = []
        low_outputs = []
        
        # Collect outputs for each input combination
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            expected_high = self.truth_table[input_state]
            
            self.send_heater_values(current_config)
            time.sleep(0.20)
            
            output = self.measure_output()
            if output is None:
                return -1000
            
            # Collect outputs based on expected state
            if expected_high:
                high_outputs.append(output)
            else:
                low_outputs.append(output)
        
        # If we have missing state combinations, penalize
        if not high_outputs or not low_outputs:
            return -500
        
        # 1. CONSISTENCY SCORES (unchanged - these are already threshold-independent)
        if len(high_outputs) > 1:
            avg_high = sum(high_outputs) / len(high_outputs)
            high_variance = sum((o - avg_high)**2 for o in high_outputs) / len(high_outputs)
            component_scores['high_consistency'] = 1.0 / (1.0 + (high_variance / 0.25))
        else:
            component_scores['high_consistency'] = 1.0
        
        if len(low_outputs) > 1:
            avg_low = sum(low_outputs) / len(low_outputs)
            low_variance = sum((o - avg_low)**2 for o in low_outputs) / len(low_outputs)
            component_scores['low_consistency'] = 1.0 / (1.0 + (low_variance / 0.25))
        else:
            component_scores['low_consistency'] = 1.0
        
        # 2. SEPARATION SCORE (completely threshold-independent)
        avg_high = sum(high_outputs) / len(high_outputs)
        avg_low = sum(low_outputs) / len(low_outputs)
        separation = avg_high - avg_low
        
        # Normalize separation by the total voltage range actually used
        all_outputs = high_outputs + low_outputs
        voltage_range_used = max(all_outputs) - min(all_outputs)
        
        if voltage_range_used > 0:
            normalized_separation = abs(separation) / voltage_range_used
            component_scores['separation'] = min(1.0, normalized_separation * 2)  # Scale to make 50% range = 1.0
        else:
            component_scores['separation'] = 0.0
        
        # 3. SUCCESS BONUS (threshold-independent ranking)
        # Check if high outputs are generally higher than low outputs
        success_count = 0
        for high_out in high_outputs:
            for low_out in low_outputs:
                if high_out > low_out:
                    success_count += 1
        
        total_comparisons = len(high_outputs) * len(low_outputs)
        success_rate = success_count / total_comparisons if total_comparisons > 0 else 0
        component_scores['success_bonus'] = success_rate
        
        # Calculate final score (0-100)
        final_score = sum(component_scores[k] * WEIGHTS[k] * 100 for k in WEIGHTS)
        
        # Optional: Add debug info for best scores
        if final_score > self.best_score:
            print(f"  Separation: {separation:.3f}V, Range: {voltage_range_used:.3f}V")
            print(f"  High avg: {avg_high:.3f}V, Low avg: {avg_low:.3f}V")
            print(f"  Success rate: {success_rate:.2%}")
        
        return final_score

    def initial_sampling(self, n_samples=20):
        """Initial sampling using Latin Hypercube Sampling for better space coverage"""
        print(f"Performing Latin Hypercube sampling with {n_samples} configurations...")
        
        # Zero configuration (baseline)
        zero_config = {h: 0.1 for h in MODIFIABLE_HEATERS}
        
        score = self.evaluate_configuration(zero_config)
        print(f"Zero configuration: Score = {score:.2f}")
        
        # Update best if this is better
        if score > self.best_score:
            self.best_score = score
            self.best_config = zero_config.copy()
            print(f"\nNew best score: {score:.2f}")
        
        # Setup for Latin Hypercube Sampling
        n_dims = len(MODIFIABLE_HEATERS)
        sampler = qmc.LatinHypercube(d=n_dims, seed=42)
        samples = sampler.random(n=n_samples-1)
        
        # Evaluate each Latin Hypercube sample
        for i, sample in enumerate(samples):
            # Convert unit hypercube to voltage range
            config = {}
            for j, h in enumerate(MODIFIABLE_HEATERS):
                config[h] = V_MIN + sample[j] * (V_MAX - V_MIN)
            
            # Evaluate configuration
            score = self.evaluate_configuration(config)
            
            # Update best if this is better
            if score > self.best_score:
                self.best_score = score
                self.best_config = config.copy()
                print(f"\nNew best score: {score:.2f}")
            
            print(f"LHS configuration {i+1}/{n_samples-1}: Score = {score:.2f}")

    def explore_around_best(self, n_samples=5):
        """Local exploration around best configuration"""
        if not self.best_config:
            print("No best configuration available for local exploration.")
            return
            
        base_config = self.best_config.copy()
        
        for i in range(n_samples):
            # Create perturbed version of best config
            new_config = base_config.copy()
            
            # Perturb random subset of heaters with small continuous steps
            for h in random.sample(MODIFIABLE_HEATERS, max(1, len(MODIFIABLE_HEATERS) // 3)):
                current = new_config.get(h, 0.1)
                # Small random perturbation (±0.3V max)
                perturbation = random.uniform(-0.3, 0.3)
                new_config[h] = max(0.1, min(4.9, current + perturbation))
            
            # Evaluate
            score = self.evaluate_configuration(new_config)
            
            # Update best if this is better
            if score > self.best_score:
                self.best_score = score
                self.best_config = new_config.copy()
                print(f"\nNew best score: {score:.2f}")
            
            print(f"Local exploration {i+1}/{n_samples}: Score = {score:.2f}") 
    
    def spsa_optimize(self, iterations=50, a=1.0, c=0.1, alpha=0.35, gamma=0.101):
        """
        - a: Initial step size originally 0.602, modified becasue it was too fast
        - c: Perturbation size
        - alpha: Step size decay parameter
        - gamma: Perturbation decay parameter
        """
        # Use best configuration if available, otherwise initialize randomly
        if self.best_config:
            theta = {h: self.best_config.get(h, 0.1) for h in MODIFIABLE_HEATERS}
        else:
            theta = {h: random.uniform(0.1, 4.9) for h in MODIFIABLE_HEATERS}
        
        # Track the heater keys for consistent ordering
        heater_keys = sorted(MODIFIABLE_HEATERS)
        
        iterations_without_improvement = 0

        # Run SPSA iterations
        for k in range(1, iterations + 1):
            # Update step size and perturbation using effective_k
            ak = a / ( k ** alpha)
            # ck = c / ( k ** gamma)
            #  Increasing minimum perturbation size
            ck = max(0.2,  c / ( k ** gamma))

            # Generate random perturbation vector (±1 for each dimension)
            delta = {h: 1 if random.random() > 0.5 else -1 for h in heater_keys}
            
            # Create perturbed configurations
            theta_plus = {h: max(0.1, min(4.9, theta[h] + ck * delta[h])) for h in heater_keys}
            theta_minus = {h: max(0.1, min(4.9, theta[h] - ck * delta[h])) for h in heater_keys}
      
            # Evaluate both configurations
            y_plus = self.evaluate_configuration(theta_plus)
            y_minus = self.evaluate_configuration(theta_minus)

            # Estimate gradient
            g_hat = {h: (y_plus - y_minus) / (2 * ck * delta[h]) for h in heater_keys}
            
            # Create new theta using gradient
            theta_new = {h: theta[h] + ak * g_hat[h] for h in heater_keys}
            
            # Ensure values stay within bounds
            theta_new = {h: max(0.1, min(4.9, v)) for h, v in theta_new.items()}

            # Evaluate new configuration
            score = self.evaluate_configuration(theta_new)

            # Update based on performance
            if score > self.best_score:
                # If better than global best, continue from new point
                self.best_score = score
                self.best_config = theta_new.copy()
                print(f'New best score: {score:.2f}')
                theta = theta_new.copy()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

                if iterations_without_improvement >= 25:
                    print(f'\nNo improvement for {iterations_without_improvement} iterations. Restarting from best\n')
                    # Reset to best configuration
                    theta = {h: self.best_config.get(h, 0.1) for h in MODIFIABLE_HEATERS}
                    # Reset effective iteration counter to get larger steps
                    #effective_k = max(1, effective_k // 2)
                    iterations_without_improvement = 0
                else:
                    # Otherwise continue from new point
                    theta = theta_new.copy()
            
            print(f'Iteration: {k} Step size {ak:.2f} Perturbation size {ck:.2f} Best {self.best_score:.2f} Score {score:.2f}')

    def test_final_configuration(self):
        if not self.best_config:
            print("No configuration available to test.")
            return False
        
        config = self.best_config
        print(f"\nTesting final {self.gate_type} gate configuration:")

        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            output_value = self.measure_output()

            print(f"\nInputs (A, B): {input_state}")
            print(f"{self.gate_type} Output: {output_value:.4f}V")

        return None
    
    def format_config(self):
        if not self.best_config:
            return {}
        
        # Create complete configuration with all heaters
        complete_config = {i: 0.0 for i in range(40)}
        
        # Add values from best configuration
        for heater, value in self.best_config.items():
            complete_config[heater] = value

        # Add fixed first layer values for completeness
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                complete_config[h] = 0.01
                
        # Format all values to 2 decimal places
        return {k: round(float(v), 2) for k, v in complete_config.items()}
    
    def cleanup(self):
        self.serial.close()
        self.scope.close()
        print("Connections closed.")
    
    def optimize(self):
        print(f"{self.gate_type} gate optimization...")

        self.initial_sampling(n_samples=75)
        self.explore_around_best(n_samples=10)
         
        self.spsa_optimize(iterations=30, a=1.5, c=1, alpha=0.3, gamma=0.15)
        self.explore_around_best(n_samples=10)
        
        self.spsa_optimize(iterations=20, a=0.8, c=0.6, alpha=0.3, gamma=0.15)
        self.explore_around_best(n_samples=10)

        self.test_final_configuration() # Test and print final results
       
        # print("\nOptimization complete!")
        # print(f"Best score: {self.best_score:.2f}")
        print("\nFinal heater configuration:")
        print(self.format_config())
        
        return self.best_config, self.best_score


def main():
    # Create optimizer for the selected gate type
    start_time = time.time()

    optimizer = SPSALogicGateOptimizer(GATE_TYPE)
    
    try:
        # Run optimization
        optimizer.optimize()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        optimizer.cleanup()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Execution time: {execution_time/60:.2f} minutes")


if __name__ == "__main__":
    main()