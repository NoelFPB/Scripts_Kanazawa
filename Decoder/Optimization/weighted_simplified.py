import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc

# Serial port configuration 
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# === DECODER CONFIGURATION ===
INPUT_PINS = [36, 37]  # Input pins (A, B)

# Voltage definitions
V_MIN = 0.1     # Representing logical LOW
V_MAX = 4.9    # Representing logical HIGH
LOW_THRESHOLD = 2   # Outputs below this are considered LOW (only for input determination)
HIGH_THRESHOLD = 3.5  # Outputs above this are considered HIGH (only for final testing)

# Heater configuration
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in INPUT_PINS]

# Test configurations for decoder
TEST_CONFIGURATIONS = [
    (V_MIN, V_MIN),    # Input 00 (output 0 should be high, others low)
    (V_MIN, V_MAX),   # Input 01 (output 1 should be high, others low)
    (V_MAX, V_MIN),   # Input 10 (output 2 should be high, others low)
    (V_MAX, V_MAX)   # Input 11 (output 3 should be high, others low)
]


class SPSADecoderOptimizer:
    """
    Optimize a physical decoder using Simultaneous Perturbation Stochastic Approximation (SPSA).
    """
    def __init__(self):
        print("Initializing decoder optimization...")
        
         # Add optimization phase tracking
        self.optimization_phase = "exploration"  # Can be: "exploration", "spsa", "testing"

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
            if h not in INPUT_PINS:
                self.base_config[h] = 0.01
        
    def _init_scope(self):
        """Initialize oscilloscope for decoder output measurement"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000

        # Setup all 4 channels for decoder output measurement
        for channel_num in range(1, 5):
            scope.write(f':CHANnel{channel_num}:DISPlay ON')
            scope.write(f':CHANnel{channel_num}:SCALe 2')
            scope.write(f':CHANnel{channel_num}:OFFSet -6')
            
        return scope
    
    def send_heater_values(self, config):
        """Send heater voltage configuration to hardware"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.serial.write(voltage_message.encode())
        self.serial.flush()
        time.sleep(0.01)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
    
    def measure_outputs(self):
        """Measure all 4 decoder output voltages from oscilloscope"""
        try:
            outputs = []
            for channel in range(1, 5):
                value = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                outputs.append(round(value, 5))
            return outputs
        except Exception as e:
            print(f"Measurement error: {e}")
            return [None] * 4

    def evaluate_configuration(self, config):
        """Evaluate decoder configuration focusing purely on separation and consistency, independent of thresholds."""
        
        # Component weights (sum to 1.0) - Only focus on what matters for separation
        WEIGHTS = {
            'high_consistency': 0.2,  # Consistency of HIGH outputs
            'low_consistency': 0.2,   # Consistency of LOW outputs  
            'separation': 0.5,         # Primary: separation between HIGH/LOW
            'success_bonus': 0.1       # Bonus for correct relative ordering
        }
        
        # Initialize scores
        component_scores = {k: 0.0 for k in WEIGHTS}
        
        # Collection of all high and low outputs for consistency calculation
        all_high_outputs = []  # Active outputs (should be HIGH)
        all_low_outputs = []   # Inactive outputs (should be LOW)
        
        # Test each input combination
        for test_config in TEST_CONFIGURATIONS:
            input_a, input_b = test_config
            
            # Determine expected active output (only use threshold for input determination)
            expected_output_idx = (1 if input_b > LOW_THRESHOLD else 0) + 2 * (1 if input_a > LOW_THRESHOLD else 0)
            
            # Configure and measure
            current_config = config.copy()
            current_config[INPUT_PINS[0]] = input_a
            current_config[INPUT_PINS[1]] = input_b
            self.send_heater_values(current_config)
            time.sleep(0.20)
            
            outputs = self.measure_outputs()
            if None in outputs:
                return -1000
            
            # Get active and inactive outputs
            active_output = outputs[expected_output_idx]
            inactive_outputs = [out for i, out in enumerate(outputs) if i != expected_output_idx]
            
            # Collect outputs for consistency calculation
            all_high_outputs.append(active_output)
            all_low_outputs.extend(inactive_outputs)
        
        # If we have missing state combinations, penalize
        if not all_high_outputs or not all_low_outputs:
            return -500
        
        
        # Normalize separation by the total voltage range actually used
        all_outputs = all_high_outputs + all_low_outputs
        voltage_range_used = max(all_outputs) - min(all_outputs)
        
        # 2. SEPARATION SCORE (completely threshold-independent)
        high_avg = sum(all_high_outputs) / len(all_high_outputs)
        low_avg = sum(all_low_outputs) / len(all_low_outputs)
        separation = high_avg - low_avg

        if separation <=0:
            if self.optimization_phase == 'exploration':
                return -100
            elif self.optimization_phase == 'SPSA':
                  if voltage_range_used > 0:
                    # Smooth penalty that preserves gradient
                    normalized_neg_separation = abs(separation) / voltage_range_used
                    penalty_score = -20 - (normalized_neg_separation * 30)  # Range: -20 to -50
                    
                    # Add small consistency bonus to maintain some gradient variation
                    if len(all_high_outputs) > 1:
                        high_variance = sum((o - high_avg)**2 for o in all_high_outputs) / len(all_high_outputs)
                        consistency_bonus = 10 / (1.0 + (high_variance / 0.25))
                        penalty_score += consistency_bonus
                    
                    return penalty_score
                  else:
                      return -100
                
            else:
                return -100
            
        #CONSISTENCY SCORES (threshold-independent)
        if len(all_high_outputs) > 1:
            high_avg = sum(all_high_outputs) / len(all_high_outputs)
            high_variance = sum((o - high_avg)**2 for o in all_high_outputs) / len(all_high_outputs)
            component_scores['high_consistency'] = 1.0 / (1.0 + (high_variance / 0.25))
        else:
            component_scores['high_consistency'] = 1.0
        
        if len(all_low_outputs) > 1:
            low_avg = sum(all_low_outputs) / len(all_low_outputs)
            low_variance = sum((o - low_avg)**2 for o in all_low_outputs) / len(all_low_outputs)
            component_scores['low_consistency'] = 1.0 / (1.0 + (low_variance / 0.25))
        else:
            component_scores['low_consistency'] = 1.0

        # Separation score

        if voltage_range_used > 0:
            normalized_separation = abs(separation) / voltage_range_used
            component_scores['separation'] = min(1.0, normalized_separation * 2)  # Scale to make 50% range = 1.0
        else:
            component_scores['separation'] = 0.0
        
        # 3. SUCCESS BONUS (threshold-independent ranking)
        # Check if high outputs are generally higher than low outputs
        success_count = 0
        for high_out in all_high_outputs:
            for low_out in all_low_outputs:
                if high_out > low_out:
                    success_count += 1
        
        total_comparisons = len(all_high_outputs) * len(all_low_outputs)
        success_rate = success_count / total_comparisons if total_comparisons > 0 else 0
        component_scores['success_bonus'] = success_rate
        
        final_score = sum(component_scores[k] * WEIGHTS[k] * 100 for k in WEIGHTS)
        
        # Optional: Add debug info for best scores
        if final_score > self.best_score:
            print(f"  Separation: {separation:.3f}V, Range: {voltage_range_used:.3f}V")
            print(f"  High avg: {high_avg:.3f}V, Low avg: {low_avg:.3f}V")
            print(f"  Success rate: {success_rate:.2%}")
        
        return final_score

    def initial_sampling(self, n_samples=20):
        """Initial sampling using Latin Hypercube Sampling for better space coverage"""
        
        self.optimization_phase = 'exploration'
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
        self.optimization_phase = 'exploration'
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
        SPSA optimization for decoder
        - a: Initial step size
        - c: Perturbation size
        - alpha: Step size decay parameter
        - gamma: Perturbation decay parameter
        """

        self.optimization_phase = 'SPSA'
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
            # Update step size and perturbation
            ak = a / (k ** alpha)
            ck = max(0.2, c / (k ** gamma))

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

                if iterations_without_improvement >= 50:
                    print(f'\nNo improvement for {iterations_without_improvement} iterations. Restarting from best\n')
                    # Reset to best configuration
                    theta = {h: self.best_config.get(h, 0.1) for h in MODIFIABLE_HEATERS}
                    iterations_without_improvement = 0
                else:
                    # Otherwise continue from new point
                    theta = theta_new.copy()
            
            print(f'Iteration: {k} Step size {ak:.2f} Perturbation size {ck:.2f} Best {self.best_score:.2f} Score {score:.2f}')


    def test_final_configuration(self):
        """Display the final decoder configuration outputs"""
        if not self.best_config:
            print("No configuration available to test.")
            return False
            
        config = self.best_config
        print(f"\nFinal decoder configuration outputs:")
        
        for test_idx, test_config in enumerate(TEST_CONFIGURATIONS):
            input_a, input_b = test_config
            
            # Determine which output should be active based on input pattern (no thresholds)
            # Pattern: 00->Ch1, 01->Ch2, 10->Ch3, 11->Ch4
            input_pattern = f"{int(input_a == V_MAX)}{int(input_b == V_MAX)}"
            expected_output_idx = test_idx  # Since TEST_CONFIGURATIONS is in order: 00, 01, 10, 11
            
            # Set up test configuration
            current_config = config.copy()
            current_config[INPUT_PINS[0]] = input_a
            current_config[INPUT_PINS[1]] = input_b
            
            # Send configuration and measure
            self.send_heater_values(current_config)
            time.sleep(0.2)
            outputs = self.measure_outputs()
            
            if None in outputs:
                print(f"Measurement error in test case {test_idx + 1}")
                continue
            
            # Display results
            print(f"\nTest Case {test_idx + 1}:")
            print(f"  Inputs (A, B): {input_a}V, {input_b}V")
            print(f"  Input Pattern: {input_pattern}")
            print(f"  Expected Active: Channel {expected_output_idx + 1}")
            print(f"  Outputs:")
            for i, out in enumerate(outputs):
                marker = " ←" if i == expected_output_idx else ""
                print(f"    Channel {i+1}: {out:.4f}V{marker}")
        
        return True

    def format_config(self):
        """Format configuration for display"""
        if not self.best_config:
            return {}
        
        # Create complete configuration with all heaters
        complete_config = {i: 0.0 for i in range(40)}
        
        # Add values from best configuration
        for heater, value in self.best_config.items():
            complete_config[heater] = value
            
        # Add fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_PINS:
                complete_config[h] = 0.01
                
        # Format all values to 2 decimal places
        return {k: round(float(v), 2) for k, v in complete_config.items()}
    
    def cleanup(self):
        """Close connections"""
        self.serial.close()
        self.scope.close()
        print("Connections closed.")
    
    def optimize(self):
        """Run multi-stage optimization for decoder"""
        print("Starting decoder optimization...")
        self.initial_sampling(n_samples=50)

        self.explore_around_best(n_samples=10)
        self.spsa_optimize(iterations=30, a=1.5, c=1, alpha=0.3, gamma=0.15)

        self.explore_around_best(n_samples=10)
        self.spsa_optimize(iterations=30, a=0.8, c=0.6, alpha=0.3, gamma=0.15)

        # Test and print final results
        self.test_final_configuration()
        
        print("\nOptimization complete!")
        print(f"Best score: {self.best_score:.2f}")
        print("\nFinal heater configuration:")
        print(self.format_config())
        
        return self.best_config, self.best_score


def main():
    # Create optimizer for decoder
    start_time = time.time()

    optimizer = SPSADecoderOptimizer()
    
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