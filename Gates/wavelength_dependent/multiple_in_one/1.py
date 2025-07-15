import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Serial port configuration 
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# Voltage definitions
V_MIN = 0.1     # Representing logical LOW
V_MAX = 4.9    # Representing logical HIGH

# === MULTI-GATE CONFIGURATION ===
# Define each logic gate you want to optimize.
# Each gate needs:
#   - name: A unique identifier for the gate.
#   - type: The logic gate type (e.g., "OR", "AND", "NAND", "NOR", "XOR", "XNOR").
#   - input_heaters: A list of heater numbers that serve as inputs for this gate.
#                    These must be within the FIXED_FIRST_LAYER range (33-40) and distinct.
#   - output_channel: The oscilloscope channel where this gate's output will be measured.
#                     These must be unique for each gate.
GATE_CONFIGURATIONS = [
    {
        "name": "OR_GATE_1",
        "type": "OR",
        "input_heaters": [34, 35],  # Heaters for input A and B
        "output_channel": 2         # Oscilloscope channel for this gate's output
    },
    {
        "name": "AND_GATE_1",
        "type": "AND",
        "input_heaters": [36, 37],
        "output_channel": 3
    },
    {
        "name": "NOR_GATE_1",
        "type": "NOR",
        "input_heaters": [38, 39],
        "output_channel": 4
    }
    # Add more gates here as needed, ensuring unique input_heaters (if they are truly independent)
    # and unique output_channel for each.
]

# Heater configuration for the entire chip
# Heaters 33 to 40 are considered part of the "first layer" for this setup.
# Any heaters in this range not used as inputs will be set to a fixed low value.
# Heaters 0 to 32 (excluding any used as inputs) will be considered "modifiable" by the optimizer.
FIXED_FIRST_LAYER_RANGE = list(range(33, 40)) # Heaters from 33 to 39 (inclusive)

def generate_truth_table(gate_type, num_inputs, v_min, v_max):
    """Generate truth table for given gate type and number of inputs."""
    
    # Dynamically generate input combinations
    input_combinations = []
    for i in range(2**num_inputs):
        binary_representation = bin(i)[2:].zfill(num_inputs)
        combination = tuple(v_max if bit == '1' else v_min for bit in binary_representation)
        input_combinations.append(combination)

    truth_tables_base = {
        "AND": [False] * (2**num_inputs - 1) + [True],
        "OR": [False] + [True] * (2**num_inputs - 1),
        "NAND": [True] * (2**num_inputs - 1) + [False],
        "NOR": [True] + [False] * (2**num_inputs - 1),
    }

    if gate_type in truth_tables_base:
        truth_values = truth_tables_base[gate_type]
    elif gate_type == "XOR":
        xor_results = []
        for combo in input_combinations:
            num_high_inputs = sum(1 for val in combo if val == v_max)
            xor_results.append(num_high_inputs % 2 == 1)
        truth_values = xor_results
    elif gate_type == "XNOR":
        xnor_results = []
        for combo in input_combinations:
            num_high_inputs = sum(1 for val in combo if val == v_max)
            xnor_results.append(num_high_inputs % 2 == 0)
        truth_values = xnor_results
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")
    
    if len(input_combinations) != len(truth_values):
        raise ValueError(f"Truth table length mismatch for {gate_type} with {num_inputs} inputs. Expected {len(input_combinations)}, got {len(truth_values)}")

    return {input_pair: output for input_pair, output in
            zip(input_combinations, truth_values)}

class BayesianLogicGateOptimizer:
    """
    Optimize multiple physical logic gates simultaneously using Bayesian Optimization with Gaussian Processes.
    Each gate has its own inputs and output, and the overall score is an aggregation of individual gate performances.
    """
    def __init__(self, gate_configurations):
        self.gate_configurations = gate_configurations
        self.gates_data = {} # Stores truth tables, input heaters, and output channels for each gate

        all_input_heaters_set = set()
        
        # Pre-process each gate configuration
        for gate_info in self.gate_configurations:
            gate_name = gate_info["name"]
            gate_type = gate_info["type"]
            input_heaters = gate_info["input_heaters"]
            output_channel = gate_info["output_channel"]
            
            num_inputs = len(input_heaters)
            truth_table = generate_truth_table(gate_type, num_inputs, V_MIN, V_MAX)
            
            self.gates_data[gate_name] = {
                "type": gate_type,
                "input_heaters": input_heaters,
                "output_channel": output_channel,
                "truth_table": truth_table,
                "num_inputs": num_inputs
            }
            all_input_heaters_set.update(input_heaters)
            print(f"Configuring {gate_name} ({gate_type} gate) with {num_inputs} inputs on Ch{output_channel}:")
            for inputs, output in truth_table.items():
                print(f"  {inputs} -> {'HIGH' if output else 'LOW'}")

        # Determine all unique modifiable heaters across all gates
        # Modifiable heaters are those not used as inputs and not explicitly fixed low in the first layer
        all_heater_ids = set(range(40))
        fixed_low_heaters_in_first_layer = set(FIXED_FIRST_LAYER_RANGE) - all_input_heaters_set
        
        # Heaters 0-32, excluding any used as inputs
        self.MODIFIABLE_HEATERS = sorted(list(
            (all_heater_ids - all_input_heaters_set) - fixed_low_heaters_in_first_layer
        ))

        # Base configuration contains heaters that are fixed (e.g., unused first layer heaters)
        self.base_config = {h: 0.01 for h in fixed_low_heaters_in_first_layer}
        
        print(f"\nTotal input heaters used: {sorted(list(all_input_heaters_set))}")
        print(f"Heaters fixed to 0.01V (unused first layer): {sorted(list(fixed_low_heaters_in_first_layer))}")
        print(f"Heaters optimized by Bayesian search: {self.MODIFIABLE_HEATERS} (total {len(self.MODIFIABLE_HEATERS)})")

        # Bayesian optimization storage
        self.X_evaluated = []  # Configuration vectors (values for MODIFIABLE_HEATERS)
        self.y_evaluated = []  # Scores
        self.gp = None         # Gaussian Process model
        
        # Initialize hardware connections
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1) # Give serial connection time to establish
        
        # Best configuration found
        self.best_config = None
        self.best_score = float('-inf')

    def _init_scope(self):
        """Initialize oscilloscope for all logic gate output measurements."""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found. Is VISA installed and your oscilloscope connected?")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
        
        # Setup all required channels based on GATE_CONFIGURATIONS
        for gate_info in self.gate_configurations:
            channel = gate_info["output_channel"]
            print(f"Setting up oscilloscope Channel {channel} for {gate_info['name']}...")
            scope.write(f':CHANnel{channel}:DISPlay ON')
            scope.write(f':CHANnel{channel}:SCALe 2') # Adjust scale as needed (e.g., 2V/div)
            scope.write(f':CHANnel{channel}:OFFSet -6') # Adjust offset to center waveform (e.g., -6V for 0-5V signals)
        
        return scope

    def config_to_array(self, config):
        """Convert a complete heater config dict to a numpy array for GP, using MODIFIABLE_HEATERS."""
        return np.array([config.get(h, 0.0) for h in self.MODIFIABLE_HEATERS])

    def array_to_config(self, x):
        """Convert a numpy array (for MODIFIABLE_HEATERS) to a partial config dict."""
        return {h: x[i] for i, h in enumerate(self.MODIFIABLE_HEATERS)}

    def send_heater_values(self, full_config):
        """Send a complete heater voltage configuration to hardware."""
        # Ensure all 40 heaters are in the message, even if some are 0.0.
        # This prevents issues if the hardware expects a fixed number of commands.
        voltage_message_parts = []
        for i in range(40):
            value = full_config.get(i, 0.0) # Default to 0.0 if heater not in config
            voltage_message_parts.append(f"{i},{value}")
        voltage_message = ";".join(voltage_message_parts) + '\n'

        self.serial.write(voltage_message.encode())
        self.serial.flush()
        time.sleep(0.01) # Short delay for serial communication
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def measure_output(self, channel):
        """Measure the logic gate output voltage from a specified oscilloscope channel."""
        try:
            # Query the VMAX (peak voltage) from the specified channel
            value = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
            return round(value, 5) # Round to 5 decimal places for consistency
        except pyvisa.errors.VisaIOError as e:
            print(f"VISA IO Error on Channel {channel}: {e}. Retrying...")
            time.sleep(0.5) # Wait a bit and try again
            try:
                value = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                return round(value, 5)
            except Exception as e_retry:
                print(f"Measurement error on Channel {channel} after retry: {e_retry}")
                return None
        except Exception as e:
            print(f"Measurement error on Channel {channel}: {e}")
            return None

    def evaluate_configuration(self, modifiable_heater_config):
        """
        Evaluates a given configuration by applying it to all gates and calculating
        an overall score based on the aggregate performance.
        """
        overall_score_sum = 0
        
        # Combine modifiable heaters with fixed base heaters
        full_heater_config = self.base_config.copy()
        full_heater_config.update(modifiable_heater_config)

        for gate_name, gate_data in self.gates_data.items():
            gate_type = gate_data["type"]
            input_heaters = gate_data["input_heaters"]
            output_channel = gate_data["output_channel"]
            truth_table = gate_data["truth_table"]
            input_combinations = list(truth_table.keys())
            
            high_outputs = []
            low_outputs = []
            all_outputs = []
            
            for input_state in input_combinations:
                current_full_config_for_gate = full_heater_config.copy()
                # Apply inputs for the current gate to the full configuration
                for i, heater in enumerate(input_heaters):
                    current_full_config_for_gate[heater] = input_state[i]
                
                expected_high = truth_table[input_state]
                
                self.send_heater_values(current_full_config_for_gate)
                time.sleep(0.20) # Time for device to stabilize
                
                output = self.measure_output(output_channel)
                if output is None:
                    print(f"Skipping evaluation due to measurement failure on Channel {output_channel}.")
                    return -1000 # Heavily penalize hardware measurement failure
                
                all_outputs.append(output)
                
                if expected_high:
                    high_outputs.append(output)
                else:
                    low_outputs.append(output)
            
            # --- Scoring for the current individual gate ---
            if not high_outputs or not low_outputs:
                # If a truth table doesn't have both HIGH and LOW expected states, this logic needs adjustment.
                # For standard gates, this implies a fundamental issue or misconfiguration.
                # For example, an OR gate must have a LOW output (0,0) and an AND gate must have a HIGH output (1,1).
                # If these lists are empty, it means the truth table itself might be trivial (e.g., all True or all False).
                # A more robust check might be `if len(high_outputs) == 0 or len(low_outputs) == 0:`
                print(f"Warning: Gate '{gate_name}' did not produce both HIGH and LOW expected states based on truth table. Score penalized.")
                current_gate_score = -200 # Penalize if a gate doesn't produce both types of outputs it's supposed to
            else:
                min_high = min(high_outputs)
                max_low = max(low_outputs)
                logic_separation = min_high - max_low
                
                er_score = 0
                if logic_separation > 0:
                    er_linear = min_high / max(max_low, 0.001) # Avoid division by zero if max_low is zero
                    er_db = 10 * np.log10(er_linear)
                    
                    if er_db < 1: er_score = 0
                    elif er_db < 3: er_score = 20 * (er_db - 1) / 2
                    elif er_db < 5: er_score = 20 + 30 * (er_db - 3) / 2
                    elif er_db < 7: er_score = 50 + 15 * (er_db - 5) / 2
                    else: er_score = 65 + 5 * (1 - np.exp(-(er_db - 7) / 3))
                else:
                    overlap_amount = abs(logic_separation)
                    er_score = -30 * min(1.0, overlap_amount / 0.5) # Penalty for overlapping levels
                
                mean_high = sum(high_outputs) / len(high_outputs)
                strength_score = 20 * min(1.0, mean_high / V_MAX) # Scale to V_MAX
                
                high_std = np.std(high_outputs) if len(high_outputs) > 1 else 0
                low_std = np.std(low_outputs) if len(low_outputs) > 1 else 0
                avg_std = (high_std + low_std) / 2
                consistency_score = 10 * np.exp(-avg_std * 5) # Reward low standard deviation
                
                current_gate_score = er_score + strength_score + consistency_score
                
                # Extra penalty for wide variation within logic states
                if len(high_outputs) > 1:
                    high_range = max(high_outputs) - min(high_outputs)
                    current_gate_score += -20 * min(1.0, high_range / 1.0)
                if len(low_outputs) > 1:
                    low_range = max(low_outputs) - min(low_outputs)
                    current_gate_score += -20 * min(1.0, low_range / 1.0)
                    
                current_gate_score = max(-50, current_gate_score) # Cap individual gate score

            overall_score_sum += current_gate_score

            # Debug output for individual gate results if score is improving overall
            if (self.best_score == float('-inf') or current_gate_score > (self.best_score / len(self.gates_data)) - 5): # Only print if close to best or improving
                print(f"  --- {gate_name} (Ch{output_channel}) Results ---")
                if high_outputs and low_outputs and logic_separation > 0:
                    print(f"  Logic Gate Extinction Ratio: {er_db:.2f}dB")
                    print(f"  Worst-case separation: {logic_separation:.3f}V")
                else:
                    print(f"  Logic levels not separated or incomplete data.")
                print(f"  HIGH outputs: {[f'{h:.3f}V' for h in high_outputs]}")
                print(f"  LOW outputs: {[f'{l:.3f}V' for l in low_outputs]}")
                print(f"  Output pattern: {[f'{o:.3f}V' for o in all_outputs]}")
                print(f"  Expected pattern: {[('HIGH' if truth_table[k] else 'LOW') for k in input_combinations]}")
                print(f"  Individual Gate Score: {current_gate_score:.2f}")

        # The overall score is the average of all individual gate scores
        final_overall_score = overall_score_sum / len(self.gate_configurations)
        return min(100, max(-50, final_overall_score)) # Ensure overall score is within reasonable bounds

    def add_evaluation(self, config, score):
        """Add evaluation to Bayesian optimizer dataset."""
        x = self.config_to_array(config) # Convert only modifiable heaters to array
        self.X_evaluated.append(x)
        self.y_evaluated.append(score)
        
        # Update best configuration if current score is higher
        if score > self.best_score:
            self.best_score = score
            self.best_config = config.copy() # Store the modifiable heater config
            print(f"🎉 New overall best score: {score:.2f}")

    def fit_gaussian_process(self):
        """Fit Gaussian Process for Bayesian optimization."""
        if len(self.X_evaluated) < 3: # Need at least 3 points to fit a GP
            return False
        
        X = np.array(self.X_evaluated)
        y = np.array(self.y_evaluated)
        
        print(f"  Fitting GP with {len(X)} points, score range: [{y.min():.1f}, {y.max():.1f}]")
        
        # Robust kernel for photonic device optimization
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.1, 100)) *
            RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
            WhiteKernel(noise_level=0.5, noise_level_bounds=(0.01, 10.0))
        )
        
        try:
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=1e-6, # Added noise level for numerical stability
                normalize_y=True,
                random_state=42
            )
            
            self.gp.fit(X, y)
            print("  GP fitted successfully!")
            return True
            
        except Exception as e:
            print(f"  GP fitting failed: {e}")
            self.gp = None # Reset GP if fitting fails
            return False
        
    def initial_sampling(self, n_samples=20):
        """Initial sampling to explore the parameter space and find promising regions."""
        print(f"\nPhase 1: Initial sampling with {n_samples} configurations...")
        
        configs = []
        
        # Add some diverse starting patterns based on common practices or heuristics
        # These patterns are applied to the MODIFIABLE_HEATERS
        num_modifiable = len(self.MODIFIABLE_HEATERS)
        if num_modifiable == 0:
            print("No modifiable heaters defined. Skipping initial sampling.")
            return 0

        good_patterns = [
            {h: 0.1 for h in self.MODIFIABLE_HEATERS},  # All low
            {h: 1.0 for h in self.MODIFIABLE_HEATERS},  # Low-medium
            {h: 2.5 for h in self.MODIFIABLE_HEATERS},  # Medium
            {h: 4.0 for h in self.MODIFIABLE_HEATERS},  # High
            {h: 4.9 for h in self.MODIFIABLE_HEATERS},  # Max
            {h: 0.1 if i < num_modifiable // 2 else 4.9 for i, h in enumerate(self.MODIFIABLE_HEATERS)}, # Half min, half max
        ]
        # Add a few random extreme patterns
        good_patterns.extend([
            {h: random.choice([V_MIN, V_MAX]) for h in self.MODIFIABLE_HEATERS} for _ in range(min(num_modifiable, 3))
        ])

        # Add a portion of these good patterns
        for pattern in good_patterns[:min(len(good_patterns), n_samples//3)]:
            configs.append(pattern)
        
        # Use Latin Hypercube sampling for the remaining initial samples to ensure good coverage
        n_remaining = n_samples - len(configs)
        if n_remaining > 0:
            sampler = qmc.LatinHypercube(d=num_modifiable, seed=42)
            samples = sampler.random(n=n_remaining)
            
            for sample in samples:
                config = {}
                for j, h in enumerate(self.MODIFIABLE_HEATERS):
                    config[h] = V_MIN + sample[j] * (V_MAX - V_MIN) # Scale to full voltage range
                configs.append(config)
        
        # Evaluate all initial configurations
        working_configs = 0
        for i, config in enumerate(configs):
            print(f"Evaluating initial config {i+1}/{n_samples}...")
            score = self.evaluate_configuration(config)
            self.add_evaluation(config, score)
            
            if score > 10: # Consider anything >10 as "working" (heuristic threshold)
                working_configs += 1
                print(f"Config {i+1}/{n_samples}: Score = {score:.2f} ✓ (Working)")
            else:
                print(f"Config {i+1}/{n_samples}: Score = {score:.2f}")
        
        print(f"Initial sampling complete. Found {working_configs} 'working' configurations (score > 10) out of {len(self.y_evaluated)} total evaluations.")
        if working_configs == 0:
            print("WARNING: No working configurations found in initial sampling. This may indicate issues with device, connections, or voltage ranges.")
        
        return working_configs

    def bayesian_optimize(self, n_iterations=30, batch_size=3):
        """
        Bayesian optimization loop. Generates candidate configurations, predicts their scores
        using the GP, and selects the most promising ones to evaluate on hardware.
        """
        print(f"\nPhase 2 & 4: Bayesian optimization for {n_iterations} cycles, testing {batch_size} configs per cycle...")
        
        total_evaluations_this_phase = 0
        
        for iteration in range(n_iterations):
            print(f"\n--- Optimization Cycle {iteration + 1}/{n_iterations} ---")
            
            # Refit GP model at the start of each cycle with all accumulated data
            gp_fitted = self.fit_gaussian_process()

            if not gp_fitted or len(self.X_evaluated) < 5:
                print("GP not yet reliably fitted or insufficient data. Using random sampling for exploration.")
                configs_to_test = []
                for _ in range(batch_size):
                    x = np.random.uniform(V_MIN, V_MAX, len(self.MODIFIABLE_HEATERS))
                    configs_to_test.append(self.array_to_config(x))
            else:
                # Generate many candidate configurations for prediction
                num_candidates_to_generate = batch_size * 500
                print(f"Generating and predicting {num_candidates_to_generate} candidate configurations...")
                
                candidates = []
                for _ in range(num_candidates_to_generate):
                    if np.random.random() < 0.7: # 70% random exploration (diversity)
                        x = np.random.uniform(V_MIN, V_MAX, len(self.MODIFIABLE_HEATERS))
                    else: # 30% local search around best (exploitation)
                        if self.best_config:
                            base_x = self.config_to_array(self.best_config)
                            # Add Gaussian noise, clip to voltage bounds
                            noise = np.random.normal(0, 0.5, len(self.MODIFIABLE_HEATERS))
                            x = np.clip(base_x + noise, V_MIN, V_MAX)
                        else:
                            x = np.random.uniform(V_MIN, V_MAX, len(self.MODIFIABLE_HEATERS)) # Fallback if no best_config yet
                    candidates.append(x)
                
                candidates = np.array(candidates)
                
                # Predict mean and standard deviation for all candidates
                mu, sigma = self.gp.predict(candidates, return_std=True)
                
                # Calculate Upper Confidence Bound (UCB) acquisition function
                # Beta balances exploration (high sigma) and exploitation (high mu)
                if self.best_score < 50:
                    beta = 3.0 # More exploration when scores are low
                elif self.best_score < 80:
                    beta = 2.0
                else:
                    beta = 1.0 # Less exploration when near optimal
                
                print(f"Using UCB beta value: {beta:.1f}")
                ucb_scores = mu + beta * sigma
                
                # Select the top 'batch_size' candidates to actually test on hardware
                best_indices = np.argsort(ucb_scores)[-batch_size:]
                configs_to_test = [self.array_to_config(candidates[i]) for i in best_indices]
                
                print(f"Selected top {batch_size} candidates (predicted scores):")
                for i, idx in enumerate(best_indices):
                    print(f"  Candidate {i+1}: Predicted Score={mu[idx]:.1f}±{sigma[idx]:.1f}, UCB Score={ucb_scores[idx]:.1f}")
            
            # === Hardware Evaluation ===
            print(f"Testing {len(configs_to_test)} selected configurations on hardware...")
            for i, config in enumerate(configs_to_test):
                print(f"\n  Testing candidate {i+1}/{len(configs_to_test)} on hardware...")
                
                score = self.evaluate_configuration(config) # This evaluates the full multi-gate setup
                self.add_evaluation(config, score)
                total_evaluations_this_phase += 1
                
                if score > self.best_score - 0.01: # Check if it's a new best or very close
                    print(f"  Result: {score:.2f} (Overall Best So Far: {self.best_score:.2f}) 🎉")
                else:
                    print(f"  Result: {score:.2f} (Overall Best So Far: {self.best_score:.2f})")
        
        print(f"\nBayesian optimization phase complete!")
        print(f"Total hardware evaluations in this phase: {total_evaluations_this_phase}")

    def explore_around_best(self, n_samples=5, perturbation_strength=0.3):
        """
        Performs local exploration around the current best configuration to fine-tune it.
        Perturbs a random subset of modifiable heaters.
        """
        if not self.best_config:
            print("No best configuration available for local exploration. Skipping.")
            return
        
        print(f"\nPhase 3 & 5: Local exploration around best configuration with {n_samples} samples...")
        base_modifiable_config = self.best_config.copy()
        
        for i in range(n_samples):
            new_modifiable_config = base_modifiable_config.copy()
            
            # Perturb a random subset of modifiable heaters (e.g., 1/3 of them, minimum 1)
            num_to_perturb = max(1, len(self.MODIFIABLE_HEATERS) // 3)
            heaters_to_perturb = random.sample(self.MODIFIABLE_HEATERS, num_to_perturb)
            
            for h in heaters_to_perturb:
                current_val = new_modifiable_config.get(h, V_MIN) # Get current value, default to V_MIN
                # Add a uniform random perturbation
                perturbation = random.uniform(-perturbation_strength, perturbation_strength)
                new_val = current_val + perturbation
                new_modifiable_config[h] = max(V_MIN, min(V_MAX, new_val)) # Clip to voltage bounds
            
            print(f"  Evaluating local exploration sample {i+1}/{n_samples}...")
            score = self.evaluate_configuration(new_modifiable_config)
            self.add_evaluation(new_modifiable_config, score) # Add to dataset, update best if applicable
            
            print(f"  Local exploration {i+1}/{n_samples}: Score = {score:.2f} (Best so far: {self.best_score:.2f})")

    def test_final_configuration(self):
        """
        Tests and displays the detailed performance of the final optimized configuration
        for each individual logic gate.
        """
        if not self.best_config:
            print("No final configuration to test.")
            return False
        
        final_modifiable_config = self.best_config
        print(f"\n--- FINAL VALIDATION OF OPTIMIZED CONFIGURATION ---")

        # Create the full heater configuration (including fixed base values)
        full_heater_config_for_test = self.base_config.copy()
        full_heater_config_for_test.update(final_modifiable_config)

        # Send the complete best configuration once
        self.send_heater_values(full_heater_config_for_test)
        time.sleep(0.5) # Give the hardware more time to settle for final test

        for gate_name, gate_data in self.gates_data.items():
            gate_type = gate_data["type"]
            input_heaters = gate_data["input_heaters"]
            output_channel = gate_data["output_channel"]
            truth_table = gate_data["truth_table"]
            input_combinations = list(truth_table.keys())

            print(f"\n>>> Results for {gate_name} ({gate_type} gate) on Channel {output_channel}:")

            all_outputs = []
            high_outputs = []
            low_outputs = []
            
            for input_state in input_combinations:
                # Apply specific input heater values for this test case
                current_test_config = full_heater_config_for_test.copy()
                for i, heater in enumerate(input_heaters):
                    current_test_config[heater] = input_state[i]
                
                self.send_heater_values(current_test_config) # Re-send for each input combo
                time.sleep(0.20) # Small delay
                
                output_value = self.measure_output(output_channel)
                expected = truth_table[input_state]
                
                if output_value is not None:
                    all_outputs.append(output_value)
                    if expected:
                        high_outputs.append(output_value)
                    else:
                        low_outputs.append(output_value)

                    print(f"  Inputs ({', '.join(f'H{h}={input_state[i]:.2f}V' for i, h in enumerate(input_heaters))}):")
                    print(f"  Measured Output: {output_value:.4f}V (Expected: {'HIGH' if expected else 'LOW'})")
                else:
                    print(f"  Measurement failed for inputs {input_state}. Cannot report.")


            # Calculate and report gate-specific metrics
            if high_outputs and low_outputs:
                min_high = min(high_outputs)
                max_low = max(low_outputs)
                logic_separation = min_high - max_low
                
                print(f"\n  === {gate_name} Performance Summary ===")
                if logic_separation > 0:
                    logic_er_linear = min_high / max(max_low, 0.001)
                    logic_er_db = 10 * np.log10(logic_er_linear)
                    print(f"  ✅ Working Logic Gate!")
                    print(f"  Logic Gate Extinction Ratio (worst-case): {logic_er_db:.2f} dB")
                    print(f"  - Minimum HIGH output: {min_high:.4f}V")
                    print(f"  - Maximum LOW output:  {max_low:.4f}V")
                    print(f"  - Logic Separation (min HIGH - max LOW): {logic_separation:.4f}V")
                else:
                    print(f"  ❌ Overlapping Logic Levels!")
                    print(f"  - Minimum HIGH output: {min_high:.4f}V")
                    print(f"  - Maximum LOW output:  {max_low:.4f}V")
                    print(f"  - Overlap amount:      {abs(logic_separation):.4f}V")
                
                print(f"  All HIGH outputs: {[f'{h:.3f}V' for h in high_outputs]}")
                print(f"  All LOW outputs:  {[f'{l:.3f}V' for l in low_outputs]}")
                
                # Qualitative assessment
                if logic_separation > 0:
                    if logic_er_db >= 5.0:
                        print(f"  Overall assessment: 🎉 Excellent logic gate performance!")
                    elif logic_er_db >= 3.0:
                        print(f"  Overall assessment: ✅ Good logic gate performance!")
                    elif logic_er_db >= 1.0:
                        print(f"  Overall assessment: ⚠️  Marginal logic gate performance")
                    else:
                        print(f"  Overall assessment: 🔧 Poor logic gate performance (low ER)")
                else:
                    print(f"  Overall assessment: 🔴 Failed to achieve logic separation.")
            else:
                print(f"  Could not calculate full performance metrics for {gate_name} (missing expected HIGH/LOW outputs).")
        return True

    def format_config(self):
        """
        Format the final optimized heater configuration into a dictionary
        containing all 40 heaters, including fixed values and inputs.
        """
        if not self.best_config:
            return {}
        
        # Start with all heaters at 0.0, then fill in known values
        complete_config = {i: 0.0 for i in range(40)} 
        
        # Add the optimized values for the modifiable heaters
        for heater, value in self.best_config.items():
            complete_config[heater] = value

        # Add the fixed-low values for the unused first layer heaters
        for h, value in self.base_config.items():
            complete_config[h] = value

        # Note: Input heater values are only applied during evaluation loops,
        # they are not part of the 'optimized' static configuration,
        # but their IDs are excluded from MODIFIABLE_HEATERS.
        
        return {k: round(float(v), 3) for k, v in complete_config.items()}

    def cleanup(self):
        """Clean up hardware connections."""
        print("\nCleaning up hardware connections...")
        try:
            # Set all heaters to 0.0V before closing
            self.send_heater_values({i: 0.0 for i in range(40)})
            time.sleep(0.5)
            self.serial.close()
            self.scope.close()
            print("Connections closed and heaters reset.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def optimize(self):
        """Main optimization routine."""
        print("Starting multi-gate Bayesian optimization...")

        try:
            # Phase 1: Initial broad sampling to find good starting points
            self.initial_sampling(n_samples=30)
            
            # Phase 2: First round of Bayesian optimization
            self.bayesian_optimize(n_iterations=15, batch_size=3)
            
            # Phase 3: Local exploration around the best configuration found so far
            self.explore_around_best(n_samples=8, perturbation_strength=0.3)
            
            # Phase 4: Second round of Bayesian optimization for refinement         
            self.bayesian_optimize(n_iterations=20, batch_size=3)
            
            # Phase 5: Final local exploration
            self.explore_around_best(n_samples=8, perturbation_strength=0.2) # Smaller perturbations

            # Final validation and reporting
            self.test_final_configuration()
            
            print(f"\nOptimization complete!")
            print(f"Best overall score achieved: {self.best_score:.2f}")
            print(f"Total hardware evaluations: {len(self.y_evaluated)}")
            print("\nFinal optimized heater configuration (values for all 40 heaters):")
            final_config_report = self.format_config()
            print(final_config_report)
            for h_id in sorted(final_config_report.keys()):
                print(f"  Heater {h_id}: {final_config_report[h_id]}V")
            
            return self.best_config, self.best_score
            
        except Exception as e:
            print(f"An error occurred during optimization: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return None, -1
            
        finally:
            self.cleanup()

def main():
    start_time = time.time()
    
    # Instantiate the optimizer with the desired multi-gate configuration
    optimizer = BayesianLogicGateOptimizer(GATE_CONFIGURATIONS)
    
    try:
        optimizer.optimize()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"Fatal error during main execution: {e}")
    finally:
        # Ensure cleanup always happens, even if an error occurs or interrupted
        try:
            optimizer.cleanup()
        except:
            pass # Cleanup might fail if serial/scope not initialized
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal script execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()