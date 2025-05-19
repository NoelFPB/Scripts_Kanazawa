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
GATE_TYPE = "OR"  # Logic gate type to optimize (AND, OR, NAND, NOR, XOR, XNOR)
INPUT_HEATERS = [36, 37]  # Heaters for input A and B

# Voltage definitions
LOW_VOLTAGE = 0.1     # Representing logical LOW
HIGH_VOLTAGE = 4.9    # Representing logical HIGH
LOW_THRESHOLD = 1.5   # Outputs below this are considered LOW
OPTIMAL_LOW = 1.0     # Optimal LOW output
HIGH_THRESHOLD = 4.0  # Outputs above this are considered HIGH
OPTIMAL_HIGH = 4.5    # Optimal HIGH output

# Heater configuration
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in INPUT_HEATERS]

# Create voltage options with discretization
VOLTAGE_OPTIONS = [round(v, 2) for v in np.arange(0.1, 5.0, 0.1)]

# Input combinations for testing
INPUT_COMBINATIONS = [
    (LOW_VOLTAGE, LOW_VOLTAGE),
    (LOW_VOLTAGE, HIGH_VOLTAGE),
    (HIGH_VOLTAGE, LOW_VOLTAGE),
    (HIGH_VOLTAGE, HIGH_VOLTAGE)
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
        
    def _init_scope(self):
        """Initialize oscilloscope for logic gate output measurement"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
        
        # Setup Channel 1 for logic gate output measurement
        scope.write(':CHANnel1:DISPlay ON')
        scope.write(':CHANnel1:SCALe 2')
        scope.write(':CHANnel1:OFFSet -6')
        
        # Turn off other channels
        for channel_num in range(2, 5):
            scope.write(f':CHANnel{channel_num}:DISPlay OFF')
            
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
            value = float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel1'))
            return round(value, 5)
        except Exception as e:
            print(f"Measurement error: {e}")
            return None
    
    def evaluate_configuration(self, config):
        """
        Evaluate a configuration with emphasis on output state consistency.
        Returns only the final score as that's what SPSA needs.
        """        
        # Component weights (sum to 1.0)
        WEIGHTS = {
            'high_state': 0.2,      # HIGH output performance
            'low_state': 0.2,       # LOW output performance
            'high_consistency': 0.3, # Consistency of HIGH outputs
            'low_consistency': 0,  # Consistency of LOW outputs
            'separation': 0.3       # Separation between HIGH/LOW
        }
        
        # Initialize scores
        component_scores = {k: 0.0 for k in WEIGHTS}
        
        # Collect all outputs for consistency calculation
        high_outputs = []
        low_outputs = []
        success_count = 0
        
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
                
            # Count successes (correct outputs)
            actual_high = output > HIGH_THRESHOLD
            if expected_high == actual_high:
                success_count += 1
        
        # If we have missing state combinations, penalize
        if not high_outputs or not low_outputs:
            return -500
        
        # Calculate individual state scores
        for output in high_outputs:
            if output >= OPTIMAL_HIGH:
                high_score = 1.0
            elif output >= HIGH_THRESHOLD:
                high_score = 0.5 + 0.5 * ((output - HIGH_THRESHOLD) / (OPTIMAL_HIGH - HIGH_THRESHOLD))
            else:
                high_score = max(0.0, 0.5 * (output / HIGH_THRESHOLD))
            
            component_scores['high_state'] += high_score / len(high_outputs)
        
        for output in low_outputs:
            if output <= OPTIMAL_LOW:
                low_score = 1.0
            elif output <= LOW_THRESHOLD:
                low_score = 0.5 + 0.5 * ((LOW_THRESHOLD - output) / (LOW_THRESHOLD - OPTIMAL_LOW))
            else:
                low_score = max(0.0, 0.5 * (1 - min(1, (output - LOW_THRESHOLD) / 2)))
            
            component_scores['low_state'] += low_score / len(low_outputs)
        
        # Calculate consistency scores
        if len(high_outputs) > 1:
            avg_high = sum(high_outputs) / len(high_outputs)
            high_variance = sum((o - avg_high)**2 for o in high_outputs) / len(high_outputs)
            component_scores['high_consistency'] = 1.0 / (1.0 + (high_variance / 0.25))
        elif len(high_outputs) == 1:
            component_scores['high_consistency'] = 1
        
        if len(low_outputs) > 1:
            avg_low = sum(low_outputs) / len(low_outputs)
            low_variance = sum((o - avg_low)**2 for o in low_outputs) / len(low_outputs)
            component_scores['low_consistency'] = 1.0 / (1.0 + (low_variance / 0.25))
        elif len(low_outputs) == 1:
            component_scores['low_consistency'] = 1
        
        # Calculate separation between averages
        avg_high = sum(high_outputs) / len(high_outputs) if high_outputs else 0
        avg_low = sum(low_outputs) / len(low_outputs) if low_outputs else 0
        separation = avg_high - avg_low
        
        # Score separation (normalized to ideal separation)
        ideal_separation = OPTIMAL_HIGH - OPTIMAL_LOW
        component_scores['separation'] = min(1.0, separation / ideal_separation) if separation > 0 else 0.0
        
        # Calculate final score (0-100)
        final_score = sum(component_scores[k] * WEIGHTS[k] * 100 for k in WEIGHTS)
        
        # Add success bonus for completely correct configurations
        if success_count == len(INPUT_COMBINATIONS):
            final_score += 20  # Bonus for perfect behavior
        
        # Update best if improved
        if final_score > self.best_score:
            self.best_score = final_score
            self.best_config = config.copy()
            print(f"\nNew best score: {final_score:.2f}")
            print(f"Components: HIGH={component_scores['high_state']:.2f}, " 
                f"LOW={component_scores['low_state']:.2f}, "
                f"HIGH_CONS={component_scores['high_consistency']:.2f}, "
                f"LOW_CONS={component_scores['low_consistency']:.2f}, "
                f"SEP={component_scores['separation']:.2f}")
            print(f"Success count: {success_count}/{len(INPUT_COMBINATIONS)}")
            
            if high_outputs:
                print(f"HIGH outputs: avg={sum(high_outputs)/len(high_outputs):.2f}V, " 
                    f"min={min(high_outputs):.2f}V, max={max(high_outputs):.2f}V")
            if low_outputs:
                print(f"LOW outputs: avg={sum(low_outputs)/len(low_outputs):.2f}V, "
                    f"min={min(low_outputs):.2f}V, max={max(low_outputs):.2f}V")
        
        return final_score
    
    def initial_sampling(self, n_samples=20):
        """Initial sampling using Latin Hypercube Sampling for better space coverage"""
        print(f"Performing Latin Hypercube sampling with {n_samples} configurations...")
        
        # Zero configuration (baseline)
        zero_config = {h: 0.1 for h in MODIFIABLE_HEATERS}
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                zero_config[h] = 0.01
        
        score = self.evaluate_configuration(zero_config)
        print(f"Zero configuration: Score = {score:.2f}")
        
        # Setup for Latin Hypercube Sampling
        n_dims = len(MODIFIABLE_HEATERS)
        sampler = qmc.LatinHypercube(d=n_dims, seed=42)
        samples = sampler.random(n=n_samples-1)  # -1 because we already did zero config
        
        # Map samples from [0,1] to voltage ranges
        v_min, v_max = min(VOLTAGE_OPTIONS), max(VOLTAGE_OPTIONS)
        
        # Track configurations we've tried to avoid duplicates
        tried_configs = set()
        tried_configs.add(tuple((h, zero_config.get(h, 0.1)) for h in sorted(MODIFIABLE_HEATERS)))
        
        # Function to find closest allowed voltage
        def closest_voltage(v):
            return min(VOLTAGE_OPTIONS, key=lambda x: abs(x - v))
        
        # Evaluate each Latin Hypercube sample
        for i, sample in enumerate(samples):
            # Convert unit hypercube to voltage range
            voltages = v_min + sample * (v_max - v_min)
            
            # Create configuration with discretization to allowed voltages
            config = {}
            for j, h in enumerate(MODIFIABLE_HEATERS):
                config[h] = closest_voltage(voltages[j])
            
            # Add fixed first layer values
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:
                    config[h] = 0.01
            
            # Check if configuration is unique
            config_tuple = tuple((h, config.get(h, 0.1)) for h in sorted(MODIFIABLE_HEATERS))
            if config_tuple in tried_configs:
                # Simple perturbation if duplicate
                for h in random.sample(MODIFIABLE_HEATERS, 3):
                    config[h] = random.choice(VOLTAGE_OPTIONS)
                config_tuple = tuple((h, config.get(h, 0.1)) for h in sorted(MODIFIABLE_HEATERS))
            
            tried_configs.add(config_tuple)
            
            # Evaluate configuration
            score = self.evaluate_configuration(config)
            print(f"LHS configuration {i+1}/{n_samples-1}: Score = {score:.2f}")
        
        # Add some fully random configurations for exploration diversity
        n_random = max(0, n_samples // 5)  # 20% random samples
        
        print(f"Adding {n_random} purely random configurations for diversity...")
        for i in range(n_random):
            config = {h: random.choice(VOLTAGE_OPTIONS) for h in MODIFIABLE_HEATERS}
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:
                    config[h] = 0.01
            
            config_tuple = tuple((h, config.get(h, 0.1)) for h in sorted(MODIFIABLE_HEATERS))
            if config_tuple not in tried_configs:
                tried_configs.add(config_tuple)
                score = self.evaluate_configuration(config)
                print(f"Random configuration {i+1}/{n_random}: Score = {score:.2f}")
        
        # Explore more around best configuration found so far
        if self.best_config:
            print("Exploring around best configuration found...")
            self.explore_around_best(n_samples=max(3, n_samples//10))
    
    def explore_around_best(self, n_samples=5):
        """Explore the space around the best configuration found so far"""
        if not self.best_config:
            print("No best configuration available for local exploration.")
            return
            
        base_config = self.best_config.copy()
        
        for i in range(n_samples):
            # Create perturbed version of best config
            new_config = base_config.copy()
            
            # Perturb random subset of heaters (30% of modifiable ones)
            for h in random.sample(MODIFIABLE_HEATERS, max(1, len(MODIFIABLE_HEATERS) // 3)):
                current = new_config.get(h, 0.1)
                current_idx = VOLTAGE_OPTIONS.index(current) if current in VOLTAGE_OPTIONS else 0
                
                # Take small step in either direction
                max_step = min(3, len(VOLTAGE_OPTIONS) - 1)
                step = random.randint(-max_step, max_step)
                new_idx = max(0, min(len(VOLTAGE_OPTIONS) - 1, current_idx + step))
                new_config[h] = VOLTAGE_OPTIONS[new_idx]
            
            # Keep fixed first layer heaters
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:
                    new_config[h] = 0.01
            
            # Evaluate
            score = self.evaluate_configuration(new_config)
            print(f"Local exploration {i+1}/{n_samples}: Score = {score:.2f}")
    
    def spsa_optimize(self, iterations=50, a=1.0, c=0.1, alpha=0.602, gamma=0.101):
        """
        Run SPSA optimization for the logic gate.
        
        Parameters:
        - iterations: Number of iterations
        - a: Initial step size
        - c: Perturbation size
        - alpha: Step size decay parameter
        - gamma: Perturbation decay parameter
        """
        print(f"Starting SPSA optimization for {self.gate_type} gate...")
        
        # Use best configuration if available, otherwise initialize randomly
        if self.best_config:
            theta = {h: self.best_config.get(h, 0.1) for h in MODIFIABLE_HEATERS}
        else:
            theta = {h: random.choice(VOLTAGE_OPTIONS) for h in MODIFIABLE_HEATERS}
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:
                    theta[h] = 0.01
        
        # Track the heater keys for consistent ordering
        heater_keys = sorted(MODIFIABLE_HEATERS)
        
        # Discretize function to map continuous SPSA values to allowed voltages
        def discretize(values):
            return {h: min(VOLTAGE_OPTIONS, key=lambda x: abs(x - values[h])) for h in values}
        
        # Run SPSA iterations
        for k in range(1, iterations + 1):
            # Update step size and perturbation
            ak = a / (k ** alpha)
            ck = c / (k ** gamma)
            
            # Generate random perturbation vector (±1 for each dimension)
            delta = {h: 1 if random.random() > 0.5 else -1 for h in heater_keys}
            
            # Create perturbed configurations
            theta_plus = {h: theta[h] + ck * delta[h] for h in heater_keys}
            theta_minus = {h: theta[h] - ck * delta[h] for h in heater_keys}
            
            # Add fixed heaters
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:
                    theta_plus[h] = 0.01
                    theta_minus[h] = 0.01
            
            # Discretize to allowed voltage levels
            theta_plus_disc = discretize(theta_plus)
            theta_minus_disc = discretize(theta_minus)
            
            # Evaluate both configurations
            y_plus = self.evaluate_configuration(theta_plus_disc)
            y_minus = self.evaluate_configuration(theta_minus_disc)
            
            # Estimate gradient
            g_hat = {h: (y_plus - y_minus) / (2 * ck * delta[h]) for h in heater_keys}
            
            # Update parameters
            theta = {h: theta[h] + ak * g_hat[h] for h in heater_keys}
            
            # Ensure values stay within bounds
            theta = {h: max(min(v, max(VOLTAGE_OPTIONS)), min(VOLTAGE_OPTIONS)) for h, v in theta.items()}
            
            # Evaluate new configuration after discretizing
            theta_disc = discretize(theta)
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:
                    theta_disc[h] = 0.01
                    
            self.evaluate_configuration(theta_disc)
            
            # Print progress
            if k % 5 == 0 or k == 1:
                print(f"Iteration {k}/{iterations} - Best score: {self.best_score:.2f}")
    
    def test_final_configuration(self):
        """Test and print performance of the optimized configuration"""
        if not self.best_config:
            print("No configuration available to test.")
            return False
        
        config = self.best_config
        print(f"\nTesting final {self.gate_type} gate configuration:")
        all_correct = True
        
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            output_value = self.measure_output()
                        
            # Check if output is in the right range based on expected output
            expected_high = self.truth_table[input_state]
            
            if expected_high:
                is_correct = output_value > HIGH_THRESHOLD
                output_state = "HIGH" if output_value > HIGH_THRESHOLD else "LOW"
            else:
                is_correct = output_value < LOW_THRESHOLD
                output_state = "LOW" if output_value < LOW_THRESHOLD else "HIGH"

            if not is_correct:
                all_correct = False

            print(f"\nInputs (A, B): {input_state}")
            print(f"{self.gate_type} Output: {output_value:.4f}V")
            print(f"Output is: {output_state}")
            print(f"Expected: {'HIGH' if expected_high else 'LOW'}")
            print(f"Correct: {'Yes' if is_correct else 'No'}")
                    
        if all_correct:
            print(f"\nSuccess! The {self.gate_type} gate is working correctly for all input combinations.")
        else:
            print(f"\nThe {self.gate_type} gate is not working perfectly for all input combinations.")
        
        return all_correct
    
    def format_config(self):
        """Format final configuration for display"""
        if not self.best_config:
            return {}
        
        # Create complete configuration with all heaters
        complete_config = {i: 0.0 for i in range(40)}
        
        # Add values from best configuration
        for heater, value in self.best_config.items():
            complete_config[heater] = value
            
        # Set fixed first layer heaters
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                complete_config[h] = 0.01
                
        # Format all values to 2 decimal places
        return {k: round(float(v), 2) for k, v in complete_config.items()}
    
    def cleanup(self):
        """Close connections"""
        self.serial.close()
        self.scope.close()
        print("Connections closed.")
    
    def optimize(self):
        """Run multi-stage optimization for logic gate"""
        print(f"Starting {self.gate_type} gate optimization...")
        
        # Phase 1: Enhanced initial exploration using Latin Hypercube Sampling
        self.initial_sampling(n_samples=30)
        
        # Phase 2: SPSA from multiple starting points
        # First run with larger step size for exploration
        print("\nRunning SPSA with larger step size for exploration...")
        self.spsa_optimize(iterations=30, a=1.0, c=0.2)
        
        # Phase 3: Refinement with smaller step size
        print("\nRefining with smaller step size...")
        self.spsa_optimize(iterations=20, a=0.3, c=0.1)
        
        # Phase 4: Final fine-tuning
        print("\nFinal fine-tuning...")
        self.spsa_optimize(iterations=10, a=0.1, c=0.05)
        
        # Phase 5: Last exploration around best
        print("\nFinal exploration around best configuration...")
        self.explore_around_best(n_samples=5)
        
        # Test and print final results
        self.test_final_configuration()
        
        print("\nOptimization complete!")
        print(f"Best score: {self.best_score:.2f}")
        print("\nFinal heater configuration:")
        print(self.format_config())
        
        return self.best_config, self.best_score


def main():
    # Create optimizer for the selected gate type
    optimizer = SPSALogicGateOptimizer(GATE_TYPE)
    
    try:
        # Run optimization
        optimizer.optimize()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    main()