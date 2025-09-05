import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import Matern
import json 
import os

# Serial port configuration 
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# === GATE CONFIGURATION ===
GATE_TYPE = "AND"  # Logic gate type to optimize (AND, OR, NAND, NOR, XOR, XNOR)
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

def generate_truth_table(gate_type):
    """Generate truth table for given gate type"""
    truth_tables = {
        "AND": [False, False, False, True],
        "OR": [False, True, True, True],
        "NAND": [True, True, True, False],
        "NOR": [True, False, False, False],
        "XOR": [False, True, True, False],
        "XNOR": [True, False, False, True]
    }
    
    if gate_type not in truth_tables:
        raise ValueError(f"Unknown gate type: {gate_type}")
    
    return {input_pair: output for input_pair, output in 
            zip(INPUT_COMBINATIONS, truth_tables[gate_type])}

class BayesianLogicGateOptimizer:
    """
    Optimize a physical logic gate using Bayesian Optimization with Gaussian Processes.
    """
    def __init__(self, gate_type=GATE_TYPE):
        # Initialize gate type and truth table
        self.gate_type = gate_type
        self.truth_table = generate_truth_table(gate_type)
        
        print(f"Optimizing {gate_type} gate with truth table:")
        for inputs, output in self.truth_table.items():
            print(f"  {inputs} -> {'HIGH' if output else 'LOW'}")
        
        # Bayesian optimization storage
        self.X_evaluated = []  # Configuration vectors
        self.y_evaluated = []  # Scores
        self.gp = None        # Gaussian Process model
        
        # Initialize hardware connections
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Best configuration found
        self.best_config = None
        self.best_score = float('-inf')

        self.base_config = {}
        
        # Set fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                self.base_config[h] = 0.01

    def save_progress(self, iteration_tag="progress"):
        """Save evaluated results to a JSON file for progress tracking"""
        try:
            data = {
                "iteration": iteration_tag,
                "scores": self.y_evaluated,
                "best_score": self.best_score,
                "best_config": self.best_config
            }
            filename = f"gate_optimization_{self.gate_type}_{iteration_tag}.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Progress saved to {filename}")
        except Exception as e:
            print(f"    Could not save progress: {e}")


    def _init_scope(self):
        """Initialize oscilloscope for logic gate output measurement"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
        
        # Setup Channel 2 for logic gate output measurement
        scope.write(':CHANnel2:DISPlay ON')
        scope.write(':CHANnel2:SCALe 2')
        scope.write(':CHANnel2:OFFSet -6')
        
        return scope

    def config_to_array(self, config):
        """Convert config dict to numpy array"""
        return np.array([config[h] for h in MODIFIABLE_HEATERS])

    def array_to_config(self, x):
        """Convert numpy array to config dict"""
        return {h: x[i] for i, h in enumerate(MODIFIABLE_HEATERS)}

    def send_heater_values(self, config):
        """Send heater voltage configuration to hardware"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.serial.write(voltage_message.encode())
        self.serial.flush()
        time.sleep(0.01)
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
  
        # Collect all outputs for analysis
        high_outputs = []
        low_outputs = []
        all_outputs = []
        output_details = []
        
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            expected_high = self.truth_table[input_state]
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            
            output = self.measure_output()
            if output is None:
                return -1000  # Hardware failure
            
            all_outputs.append(output)
            output_details.append({
                'input': input_state,
                'output': output,
                'expected': 'HIGH' if expected_high else 'LOW'
            })
            
            if expected_high:
                high_outputs.append(output)
            else:
                low_outputs.append(output)
        
        if len(all_outputs) != 4 or not high_outputs or not low_outputs:
            return -500  # Missing measurements
        
        # === PRIMARY METRIC: Logic Gate Extinction Ratio (70 points) ===
        # Use worst-case scenario for reliable logic operation
        min_high = min(high_outputs)
        max_low = max(low_outputs)
        
        # Check for logic separation first
        logic_separation = min_high - max_low
        
        if logic_separation > 0:
            # Working logic gate - calculate meaningful extinction ratio
            er_linear = min_high / max(max_low, 0.001)
            er_db = 10 * np.log10(er_linear)
            
            # Score based on extinction ratio (targeting 3-7dB for good logic gates)
            if er_db < 1:
                er_score = 0
            elif er_db < 3:
                er_score = 20 * (er_db - 1) / 2  # Linear 0→20 for 1-3dB
            elif er_db < 5:
                er_score = 20 + 30 * (er_db - 3) / 2  # Linear 20→50 for 3-5dB
            elif er_db < 7:
                er_score = 50 + 15 * (er_db - 5) / 2  # Linear 50→65 for 5-7dB
            else:
                er_score = 65 + 5 * (1 - np.exp(-(er_db - 7) / 3))  # Saturating 65→70
        else:
            # Overlapping logic levels - penalize but provide gradient
            er_db = 0
            er_linear = 1
            overlap_amount = abs(logic_separation)
            er_score = -30 * min(1.0, overlap_amount / 0.5)  # -30 to 0 based on overlap
        
        # === SIGNAL STRENGTH (20 points) ===
        # Reward higher absolute output levels for better SNR
        mean_high = sum(high_outputs) / len(high_outputs)
        strength_score = 20 * min(1.0, mean_high / 4.0)  # Scale to 3V max
            
        # === CONSISTENCY (10 points) ===
        # Reward consistent outputs within each logic state
        high_std = np.std(high_outputs) if len(high_outputs) > 1 else 0
        low_std = np.std(low_outputs) if len(low_outputs) > 1 else 0
        avg_std = (high_std + low_std) / 2
        consistency_score = 10 * np.exp(-avg_std * 3)  # Full points for <0.2V std dev
        
        total_score = er_score + strength_score + consistency_score
        
        # Add this after calculating consistency_score:
        # Extra penalty for HIGH output variation
        if len(high_outputs) > 1:
            high_range = max(high_outputs) - min(high_outputs)
            high_consistency_penalty = -20 * min(1.0, high_range / 1.0)  # -20 points for 1V+ variation
            total_score += high_consistency_penalty

        if len(low_outputs) > 1:
            low_range = max(low_outputs) - min(low_outputs)
            low_consistency_penalty = -20 * min(1.0, low_range / 1.0)  # -20 points for 1V+ variation
            total_score += low_consistency_penalty
        

        # Cap at 100 points
        final_score = min(100, max(-50, total_score))
        
        # Debug output for improvements
        if final_score > self.best_score:
            print(f"  Logic Gate Extinction Ratio: {er_db:.2f}dB (linear: {er_linear:.1f})")
            print(f"  Worst-case separation: {logic_separation:.3f}V")
            print(f"  HIGH outputs: {[f'{h:.3f}V' for h in high_outputs]} (min: {min_high:.3f}V)")
            print(f"  LOW outputs: {[f'{l:.3f}V' for l in low_outputs]} (max: {max_low:.3f}V)")
            print(f"  Output pattern: {[f'{o:.3f}V' for o in all_outputs]}")
            print(f"  Expected pattern: {[d['expected'] for d in output_details]}")
        
        return final_score

    def add_evaluation(self, config, score):
        """Add evaluation to Bayesian optimizer dataset"""
        x = self.config_to_array(config)
        self.X_evaluated.append(x)
        self.y_evaluated.append(score)
        
        # Update best configuration
        if score > self.best_score:
            self.best_score = score
            self.best_config = config.copy()
            print(f"New best score: {score:.2f}")

    def fit_gaussian_process(self):
        """Fit Gaussian Process for Bayesian optimization"""
        if len(self.X_evaluated) < 3:
            return False
        
        X = np.array(self.X_evaluated)
        y = np.array(self.y_evaluated)
        
        print(f"    Fitting GP with {len(X)} points, score range: [{y.min():.1f}, {y.max():.1f}]")
        
        # Robust kernel for photonic device optimization
        # kernel = (
        #     ConstantKernel(1.0, constant_value_bounds=(0.1, 100)) *
        #     RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
        #     WhiteKernel(noise_level=0.5, noise_level_bounds=(0.01, 10.0))
        # )
        
        # Try a more flexible kernel:

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.1, 100.0)) * 
            Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) +
            WhiteKernel(noise_level=0.25, noise_level_bounds=(0.001, 1))
        )
        try:
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=1e-6,
                normalize_y=True,
                random_state=42
            )
            
            self.gp.fit(X, y)
            print("    GP fitted successfully!")
            return True
            
        except Exception as e:
            print(f"    GP fitting failed: {e}")
            self.gp = None
            return False
        
    def initial_sampling(self, n_samples=20):
        """Initial sampling focused on finding real working logic gates"""
        print(f"Initial sampling with {n_samples} configurations...")
        
        configs = []
        
        # More diverse starting patterns for real photonic logic gates
        good_patterns = [
            {h: 0.1 for h in MODIFIABLE_HEATERS},  # All low
            {h: 1.0 for h in MODIFIABLE_HEATERS},  # Low-medium
            {h: 2.5 for h in MODIFIABLE_HEATERS},  # Medium
            {h: 4.0 for h in MODIFIABLE_HEATERS},  # High
            {h: 4.9 for h in MODIFIABLE_HEATERS},  # Max
            {h: 0.1 if i < len(MODIFIABLE_HEATERS)//2 else 4.9 for i, h in enumerate(MODIFIABLE_HEATERS)},  # Half min, half max
            {h: random.choice([0.1, 1.0, 2.0, 3.0, 4.0, 4.9]) for h in MODIFIABLE_HEATERS},  # Random extremes
            {h: 0.1 + (i % 10) * 0.5 for i, h in enumerate(MODIFIABLE_HEATERS)},  # Wide stepped pattern
        ]
        
        # Add good patterns
        for pattern in good_patterns[:min(6, n_samples//3)]:
            configs.append(pattern)
        
        # More diverse Latin Hypercube sampling (full voltage range)
        n_remaining = n_samples - len(configs)
        if n_remaining > 0:
            sampler = qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS), seed=42)
            samples = sampler.random(n=n_remaining)
            
            for sample in samples:
                config = {}
                for j, h in enumerate(MODIFIABLE_HEATERS):
                    # Use full voltage range - we need bigger differences for real logic gates
                    config[h] = V_MIN + sample[j] * (V_MAX - V_MIN)
                configs.append(config)
        
        # Evaluate all initial configurations
        working_configs = 0
        for i, config in enumerate(configs):
            score = self.evaluate_configuration(config)
            self.add_evaluation(config, score)
            
            if score > 10:  # Consider anything >10 as "working"
                working_configs += 1
                print(f"Config {i+1}/{n_samples}: Score = {score:.2f} ✓")
            else:
                print(f"Config {i+1}/{n_samples}: Score = {score:.2f}")
        
        print(f"Initial sampling complete. Working configs: {working_configs}/{len(self.y_evaluated)}")
        if working_configs == 0:
            print("WARNING: No working configurations found! May need to adjust voltage ranges or hardware.")
        
        return working_configs

    def bayesian_optimize(self, n_iterations=30, batch_size=3):
        """
        Efficient Bayesian optimization: Generate many candidates, test only the best
        
        Args:
            n_iterations: Number of optimization cycles
            batch_size: How many configs to actually test per cycle (keep small!)
        """
        print(f"\nBayesian optimization for {n_iterations} cycles, testing {batch_size} configs per cycle...")
        
        total_evaluations = 0
        
        for iteration in range(n_iterations):
            print(f"\n--- Optimization Cycle {iteration + 1}/{n_iterations} ---")

            self.save_progress(iteration_tag=f"iter{iteration+1}")
            
            if self.gp is None or len(self.X_evaluated) < 5:
                print("Insufficient data for GP, using random sampling...")
                # Random exploration for early iterations
                configs_to_test = []
                for _ in range(batch_size):
                    x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
                    configs_to_test.append(self.array_to_config(x))
            else:
                # SMART APPROACH: Generate many candidates, predict them all, test only the best
                print(f"Generating and predicting {batch_size * 500} candidate configurations...")
                
                candidates = []
                
                # Generate diverse candidate pool
                for _ in range(batch_size * 500):  # Generate many candidates
                    if np.random.random() < 0.7:
                        # 70% random exploration
                        x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
                    else:
                        # 30% local search around best
                        if self.best_config:
                            base_x = self.config_to_array(self.best_config)
                            noise = np.random.normal(0, 0.5, len(MODIFIABLE_HEATERS))
                            x = np.clip(base_x + noise, V_MIN, V_MAX)
                        else:
                            x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
                    
                    candidates.append(x)
                
                candidates = np.array(candidates)
                
                # Predict scores for ALL candidates (this is fast!)
                mu, sigma = self.gp.predict(candidates, return_std=True)
                
                if self.best_score < 50:
                    beta = 4.0
                elif self.best_score < 60:
                    beta = 2.0
                else:
                    beta = 2.0

                print("Beta ", beta)
                ucb_scores = mu + beta * sigma
                
                # Select the top candidates to actually test
                best_indices = np.argsort(ucb_scores)[-batch_size:]
                configs_to_test = [self.array_to_config(candidates[i]) for i in best_indices]
                
                print(f"Selected top {batch_size} candidates:")
                for i, idx in enumerate(best_indices):
                    print(f"  Candidate {i+1}: Predicted={mu[idx]:.1f}±{sigma[idx]:.1f}, UCB={ucb_scores[idx]:.1f}")
            
            # NOW test only the selected configurations on hardware
            print(f"Testing {len(configs_to_test)} configurations on hardware...")
            
            for i, config in enumerate(configs_to_test):
                print(f"\n  Testing candidate {i+1}/{len(configs_to_test)}...")
                
                # FIXED: Evaluate configuration first, then check if it's a new best
                score = self.evaluate_configuration(config)
                self.add_evaluation(config, score)
                total_evaluations += 1
                
                # Show details if it's a new best
                if score > self.best_score - 0.01:  # Small tolerance for floating point
                    print(f"    🎉 NEW BEST or near-best score!")
                
                print(f"    Result: {score:.2f} (best so far: {self.best_score:.2f})")
            
            # Update GP after each cycle
            print("Updating Gaussian Process with new data...")
            self.fit_gaussian_process()
            
            # Show efficiency metrics only if we used GP predictions
            if self.gp is not None and iteration > 0 and 'mu' in locals():
                predicted_best = max(mu)
                actual_best = self.best_score
                print(f"  Efficiency: Predicted best={predicted_best:.1f}, Actual best={actual_best:.1f}")
            elif self.gp is not None:
                print(f"  GP ready for next cycle. Current best: {self.best_score:.1f}")


            if (iteration + 1) % 10 == 0:
                self.save_progress(iteration_tag=f"iter{iteration+1}")
        
        print(f"\nBayesian optimization complete!")
        print(f"Total hardware evaluations: {total_evaluations}")
        if n_iterations > 0 and batch_size > 0:
            total_candidates = n_iterations * batch_size * 500
            if total_candidates > 0:
                print(f"Efficiency: {total_evaluations}/{total_candidates} = {total_evaluations/total_candidates:.1%} of candidates tested")
        print(f"Final best score: {self.best_score:.2f}")

    def explore_around_best(self, n_samples=5):
        """Local exploration around best configuration"""
        if not self.best_config:
            print("No best configuration available for local exploration.")
            return
        
        print(f"\nLocal exploration with {n_samples} configurations...")
        base_config = self.best_config.copy()
        
        for i in range(n_samples):
            # Create perturbed version of best config
            new_config = base_config.copy()
            
            # Perturb random subset of heaters
            for h in random.sample(MODIFIABLE_HEATERS, max(1, len(MODIFIABLE_HEATERS) // 3)):
                current = new_config.get(h, 0.1)
                perturbation = random.uniform(-0.3, 0.3)
                new_config[h] = max(V_MIN, min(V_MAX, current + perturbation))
            
            # Evaluate
            score = self.evaluate_configuration(new_config)
            self.add_evaluation(new_config, score)
            
            print(f"Local exploration {i+1}/{n_samples}: Score = {score:.2f}")

    def test_final_configuration(self):
        """Test and display final configuration with logic-relevant extinction ratio"""
        if not self.best_config:
            print("No configuration available to test.")
            return False
        
        config = self.best_config
        print(f"\nTesting final {self.gate_type} gate configuration:")

        all_outputs = []
        high_outputs = []
        low_outputs = []
        
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            output_value = self.measure_output()
            expected = self.truth_table[input_state]
            
            all_outputs.append(output_value)
            if expected:
                high_outputs.append(output_value)
            else:
                low_outputs.append(output_value)

            print(f"Inputs (A, B): {input_state}")
            print(f"{self.gate_type} Output: {output_value:.4f}V (expect {'HIGH' if expected else 'LOW'})")

        # Calculate both extinction ratios
        if len(all_outputs) >= 2 and high_outputs and low_outputs:
            # Logic-relevant extinction ratio (worst-case)
            min_high = min(high_outputs)
            max_low = max(low_outputs)
            logic_separation = min_high - max_low
            
            
            print(f"\n=== PERFORMANCE METRICS ===")
            
            if logic_separation > 0:
                logic_er_linear = min_high / max_low
                logic_er_db = 10 * np.log10(logic_er_linear)
                print(f"✅ WORKING LOGIC GATE!")
                print(f"Logic Gate Extinction Ratio: {logic_er_db:.2f} dB (worst-case)")
                print(f"  - Minimum HIGH: {min_high:.4f}V")
                print(f"  - Maximum LOW:  {max_low:.4f}V")
                print(f"  - Separation:   {logic_separation:.4f}V")
            else:
                print(f"❌ OVERLAPPING LOGIC LEVELS!")
                print(f"  - Minimum HIGH: {min_high:.4f}V")
                print(f"  - Maximum LOW:  {max_low:.4f}V")
                print(f"  - Overlap:      {abs(logic_separation):.4f}V")
            
            print(f"\nHIGH outputs: {[f'{h:.3f}V' for h in high_outputs]}")
            print(f"LOW outputs:  {[f'{l:.3f}V' for l in low_outputs]}")
            
            # Performance assessment
            if logic_separation > 0:
                logic_er_db = 10 * np.log10(min_high / max_low)
                if logic_er_db >= 5.0:
                    print(f"🎉 Excellent logic gate performance!")
                elif logic_er_db >= 3.0:
                    print(f"✅ Good logic gate performance!")
                elif logic_er_db >= 1.0:
                    print(f"⚠️  Marginal logic gate performance")
                else:
                    print(f"🔧 Poor logic gate performance")

        return None

    def format_config(self):
        """Format final configuration for output"""
        if not self.best_config:
            return {}
        
        # Create complete configuration with all heaters
        complete_config = {i: 0.0 for i in range(40)}
        
        # Add values from best configuration
        for heater, value in self.best_config.items():
            complete_config[heater] = value

        # Add fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                complete_config[h] = 0.01
                
        return {k: round(float(v), 3) for k, v in complete_config.items()}

    def cleanup(self):
        """Clean up hardware connections"""
        try:
            self.serial.close()
            self.scope.close()
            print("Connections closed.")
        except:
            pass

    def optimize(self):
        """Main optimization routine"""
        print(f"Bayesian optimization for {self.gate_type} gate...")

        try:
            # Phase 1: Initial sampling
            self.initial_sampling(n_samples=20)
            
            # Phase 2: Bayesian optimization
            self.bayesian_optimize(n_iterations=20)
            
            # # Phase 3: Local exploration around best
            self.explore_around_best(n_samples=15)
            
            # # Phase 4: Final Bayesian refinement
            self.bayesian_optimize(n_iterations=20)
            
            self.explore_around_best(n_samples=8)
            
            
            # Test final configuration
            self.test_final_configuration()
            
            print(f"\nOptimization complete!")
            print(f"Best score: {self.best_score:.2f}")
            print(f"Total evaluations: {len(self.y_evaluated)}")
            print("\nFinal heater configuration:")
            print(self.format_config())
            
            return self.best_config, self.best_score
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None, -1
        
        finally:
            self.cleanup()

def main():
    start_time = time.time()
    optimizer = BayesianLogicGateOptimizer(GATE_TYPE)
    
    try:
        optimizer.optimize()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            optimizer.cleanup()
        except:
            pass
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Execution time: {execution_time/60:.2f} minutes")

if __name__ == "__main__":
    main()