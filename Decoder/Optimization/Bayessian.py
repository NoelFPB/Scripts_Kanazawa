import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Serial port configuration 
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# === DECODER CONFIGURATION ===
INPUT_PINS = [36, 37]  # Input pins (A, B)

# Voltage definitions
V_MIN = 0.1     # Representing logical LOW
V_MAX = 4.9     # Representing logical HIGH

# Heater configuration
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in INPUT_PINS]

# Test configurations for decoder
TEST_CONFIGURATIONS = [
    (V_MIN, V_MIN),    # Input 00 (output 0 should be high, others low)
    (V_MIN, V_MAX),    # Input 01 (output 1 should be high, others low)
    (V_MAX, V_MIN),    # Input 10 (output 2 should be high, others low)
    (V_MAX, V_MAX)     # Input 11 (output 3 should be high, others low)
]

class BayesianDecoderOptimizer:
    """
    Optimize a physical decoder using Bayesian Optimization with Gaussian Processes.
    """
    def __init__(self):
        print("Initializing Bayesian decoder optimization...")
        
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
        """
        Decoder scoring function optimized for Bayesian optimization.
        
        A decoder should have one HIGH output and three LOW outputs for each input combination.
        We use extinction ratio concepts adapted for multi-output devices.
        """
        
        # Collect all measurements for analysis
        all_active_outputs = []    # Outputs that should be HIGH
        all_inactive_outputs = []  # Outputs that should be LOW
        test_results = []
        
        for test_idx, test_config in enumerate(TEST_CONFIGURATIONS):
            input_a, input_b = test_config
            expected_active_idx = test_idx  # Channel that should be active
            
            # Configure inputs and measure
            current_config = config.copy()
            current_config[INPUT_PINS[0]] = input_a
            current_config[INPUT_PINS[1]] = input_b
            
            self.send_heater_values(current_config)
            time.sleep(0.20)
            
            outputs = self.measure_outputs()
            if None in outputs:
                return -1000  # Hardware failure
            
            # Separate active and inactive outputs
            active_output = outputs[expected_active_idx]
            inactive_outputs = [outputs[i] for i in range(4) if i != expected_active_idx]
            
            all_active_outputs.append(active_output)
            all_inactive_outputs.extend(inactive_outputs)
            
            test_results.append({
                'input': test_config,
                'expected_active': expected_active_idx,
                'outputs': outputs,
                'active_output': active_output,
                'inactive_outputs': inactive_outputs
            })
        
        if len(all_active_outputs) != 4 or len(all_inactive_outputs) != 12:
            return -500  # Missing measurements
        
        # === PRIMARY METRIC: Decoder Extinction Ratio (70 points) ===
        # Use worst-case scenario for reliable decoder operation
        min_active = min(all_active_outputs)
        max_inactive = max(all_inactive_outputs)
        
        # Check for logic separation
        logic_separation = min_active - max_inactive
        
        if logic_separation > 0:
            # Working decoder - calculate extinction ratio
            er_linear = min_active / max(max_inactive, 0.001)
            er_db = 10 * np.log10(er_linear)
            
            # Score based on extinction ratio (targeting 3-7dB for good decoders)
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
        
        # === SIGNAL STRENGTH (15 points) ===
        # Reward higher absolute output levels for better SNR
        mean_active = sum(all_active_outputs) / len(all_active_outputs)
        strength_score = 15 * min(1.0, mean_active / 3.0)  # Scale to 3V max
        
        # === CONSISTENCY (10 points) ===
        # Reward consistent outputs within each logic state
        active_std = np.std(all_active_outputs) if len(all_active_outputs) > 1 else 0
        inactive_std = np.std(all_inactive_outputs) if len(all_inactive_outputs) > 1 else 0
        avg_std = (active_std + inactive_std) / 2
        consistency_score = 10 * np.exp(-avg_std * 5)  # Full points for <0.2V std dev
        
        # === SELECTIVITY BONUS (5 points) ===
        # Bonus for each test case where the correct output is highest
        selectivity_score = 0
        for result in test_results:
            expected_idx = result['expected_active']
            outputs = result['outputs']
            if outputs[expected_idx] == max(outputs):
                selectivity_score += 5 / 4  # 1.25 points per correct case
        
        # === COMBINE SCORES ===
        total_score = er_score + strength_score + consistency_score + selectivity_score
        
        # Cap at 100 points
        final_score = min(100, max(-50, total_score))
        
        # Debug output for improvements
        if final_score > self.best_score:
            print(f"  Decoder Extinction Ratio: {er_db:.2f}dB (linear: {er_linear:.1f})")
            print(f"  Worst-case separation: {logic_separation:.3f}V")
            print(f"  ACTIVE outputs: {[f'{h:.3f}V' for h in all_active_outputs]} (min: {min_active:.3f}V)")
            print(f"  INACTIVE outputs: {[f'{l:.3f}V' for l in all_inactive_outputs[:4]]}... (max: {max_inactive:.3f}V)")
            print(f"  Test patterns:")
            for i, result in enumerate(test_results):
                pattern = f"{int(result['input'][0] == V_MAX)}{int(result['input'][1] == V_MAX)}"
                outputs_str = [f'{o:.3f}' for o in result['outputs']]
                print(f"    {pattern} -> Ch{result['expected_active']+1}: {outputs_str}")
        
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
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.1, 100)) *
            RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
            WhiteKernel(noise_level=0.5, noise_level_bounds=(0.01, 10.0))
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

    def acquisition_function(self, x):
        """Upper Confidence Bound acquisition function"""
        if self.gp is None:
            return random.random()
        
        x = x.reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)
        mu, sigma = mu[0], sigma[0]
        
        # UCB with exploration parameter
        beta = 3.0  # Exploration vs exploitation balance
        ucb = mu + beta * sigma
        
        return ucb

    def suggest_next_config(self):
        """Suggest next configuration using Bayesian optimization"""
        if self.gp is None or len(self.X_evaluated) < 5:
            # Random exploration for initial points
            x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
            return self.array_to_config(x)
        
        # Generate candidate configurations
        n_candidates = 3000
        
        # Mix of random and local search candidates
        candidates = []
        
        # 70% random exploration
        n_random = int(0.7 * n_candidates)
        random_candidates = np.random.uniform(V_MIN, V_MAX, size=(n_random, len(MODIFIABLE_HEATERS)))
        candidates.extend(random_candidates)
        
        # 30% local search around best configurations
        n_local = n_candidates - n_random
        if self.best_config:
            for _ in range(n_local):
                base_x = self.config_to_array(self.best_config)
                noise = np.random.normal(0, 0.5, len(MODIFIABLE_HEATERS))
                candidate = np.clip(base_x + noise, V_MIN, V_MAX)
                candidates.append(candidate)
        
        candidates = np.array(candidates)
        
        # Evaluate acquisition function
        acquisition_values = [self.acquisition_function(c) for c in candidates]
        
        # Select best candidate
        best_idx = np.argmax(acquisition_values)
        return self.array_to_config(candidates[best_idx])

    def initial_sampling(self, n_samples=20):
        """Initial sampling focused on finding working decoders"""
        print(f"Initial sampling with {n_samples} configurations...")
        
        configs = []
        
        # More diverse starting patterns for decoder optimization
        good_patterns = [
            {h: 0.1 for h in MODIFIABLE_HEATERS},  # All low
            {h: 1.0 for h in MODIFIABLE_HEATERS},  # Low-medium
            {h: 2.5 for h in MODIFIABLE_HEATERS},  # Medium
            {h: 4.0 for h in MODIFIABLE_HEATERS},  # High
            {h: 4.9 for h in MODIFIABLE_HEATERS},  # Max
            {h: 0.1 if i < len(MODIFIABLE_HEATERS)//2 else 4.9 for i, h in enumerate(MODIFIABLE_HEATERS)},  # Half-half
            {h: random.choice([0.1, 1.0, 2.0, 3.0, 4.0, 4.9]) for h in MODIFIABLE_HEATERS},  # Random levels
            {h: 0.1 + (i % 10) * 0.5 for i, h in enumerate(MODIFIABLE_HEATERS)},  # Stepped pattern
        ]
        
        # Add good patterns
        for pattern in good_patterns[:min(8, n_samples//3)]:
            configs.append(pattern)
        
        # Latin Hypercube sampling for remaining
        n_remaining = n_samples - len(configs)
        if n_remaining > 0:
            sampler = qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS), seed=42)
            samples = sampler.random(n=n_remaining)
            
            for sample in samples:
                config = {}
                for j, h in enumerate(MODIFIABLE_HEATERS):
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
        Efficient Bayesian optimization for decoder
        """
        print(f"\nBayesian optimization for {n_iterations} cycles, testing {batch_size} configs per cycle...")
        
        total_evaluations = 0
        
        for iteration in range(n_iterations):
            print(f"\n--- Optimization Cycle {iteration + 1}/{n_iterations} ---")
            
            if self.gp is None or len(self.X_evaluated) < 5:
                print("Insufficient data for GP, using random sampling...")
                configs_to_test = []
                for _ in range(batch_size):
                    x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
                    configs_to_test.append(self.array_to_config(x))
            else:
                print(f"Generating and predicting {batch_size * 500} candidate configurations...")
                
                candidates = []
                
                # Generate diverse candidate pool
                for _ in range(batch_size * 500):
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
                
                # Predict scores for ALL candidates
                mu, sigma = self.gp.predict(candidates, return_std=True)
                
                # UCB acquisition function
                beta = 3.0
                ucb_scores = mu + beta * sigma
                
                # Select the top candidates to actually test
                best_indices = np.argsort(ucb_scores)[-batch_size:]
                configs_to_test = [self.array_to_config(candidates[i]) for i in best_indices]
                
                print(f"Selected top {batch_size} candidates:")
                for i, idx in enumerate(best_indices):
                    print(f"  Candidate {i+1}: Predicted={mu[idx]:.1f}±{sigma[idx]:.1f}, UCB={ucb_scores[idx]:.1f}")
            
            # Test selected configurations on hardware
            print(f"Testing {len(configs_to_test)} configurations on hardware...")
            
            for i, config in enumerate(configs_to_test):
                print(f"\n  Testing candidate {i+1}/{len(configs_to_test)}...")
                
                score = self.evaluate_configuration(config)
                self.add_evaluation(config, score)
                total_evaluations += 1
                
                if score > self.best_score - 0.01:
                    print(f"    🎉 NEW BEST or near-best score!")
                
                print(f"    Result: {score:.2f} (best so far: {self.best_score:.2f})")
            
            # Update GP after each cycle
            print("Updating Gaussian Process with new data...")
            self.fit_gaussian_process()
            
            # Show efficiency metrics
            if self.gp is not None and iteration > 0 and 'mu' in locals():
                predicted_best = max(mu)
                actual_best = self.best_score
                print(f"  Efficiency: Predicted best={predicted_best:.1f}, Actual best={actual_best:.1f}")
            elif self.gp is not None:
                print(f"  GP ready for next cycle. Current best: {self.best_score:.1f}")
        
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
        """Test and display final decoder configuration"""
        if not self.best_config:
            print("No configuration available to test.")
            return False
        
        config = self.best_config
        print(f"\nTesting final decoder configuration:")
        
        all_active_outputs = []
        all_inactive_outputs = []
        
        for test_idx, test_config in enumerate(TEST_CONFIGURATIONS):
            input_a, input_b = test_config
            expected_active_idx = test_idx
            
            # Set up test configuration
            current_config = config.copy()
            current_config[INPUT_PINS[0]] = input_a
            current_config[INPUT_PINS[1]] = input_b
            
            # Send configuration and measure
            self.send_heater_values(current_config)
            time.sleep(0.25)
            outputs = self.measure_outputs()
            
            if None in outputs:
                print(f"Measurement error in test case {test_idx + 1}")
                continue
            
            # Collect data for analysis
            active_output = outputs[expected_active_idx]
            inactive_outputs = [outputs[i] for i in range(4) if i != expected_active_idx]
            
            all_active_outputs.append(active_output)
            all_inactive_outputs.extend(inactive_outputs)
            
            # Display results
            input_pattern = f"{int(input_a == V_MAX)}{int(input_b == V_MAX)}"
            print(f"\nInput Pattern {input_pattern} (Expected: Ch{expected_active_idx + 1} HIGH):")
            for i, out in enumerate(outputs):
                marker = " ← ACTIVE" if i == expected_active_idx else ""
                print(f"  Channel {i+1}: {out:.4f}V{marker}")
        
        # Calculate performance metrics
        if all_active_outputs and all_inactive_outputs:
            min_active = min(all_active_outputs)
            max_inactive = max(all_inactive_outputs)
            logic_separation = min_active - max_inactive
            
            print(f"\n=== DECODER PERFORMANCE METRICS ===")
            
            if logic_separation > 0:
                logic_er_linear = min_active / max_inactive
                logic_er_db = 10 * np.log10(logic_er_linear)
                print(f"✅ WORKING DECODER!")
                print(f"Decoder Extinction Ratio: {logic_er_db:.2f} dB")
                print(f"  - Minimum ACTIVE:   {min_active:.4f}V")
                print(f"  - Maximum INACTIVE: {max_inactive:.4f}V")
                print(f"  - Separation:       {logic_separation:.4f}V")
                
                if logic_er_db >= 5.0:
                    print(f"🎉 Excellent decoder performance!")
                elif logic_er_db >= 3.0:
                    print(f"✅ Good decoder performance!")
                elif logic_er_db >= 1.0:
                    print(f"⚠️  Marginal decoder performance")
                else:
                    print(f"🔧 Poor decoder performance")
            else:
                print(f"❌ OVERLAPPING LOGIC LEVELS!")
                print(f"  - Minimum ACTIVE:   {min_active:.4f}V")
                print(f"  - Maximum INACTIVE: {max_inactive:.4f}V")
                print(f"  - Overlap:          {abs(logic_separation):.4f}V")
            
            print(f"\nACTIVE outputs:   {[f'{h:.3f}V' for h in all_active_outputs]}")
            print(f"INACTIVE outputs: {[f'{l:.3f}V' for l in all_inactive_outputs[:8]]}...")
        
        return True

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
            if h not in INPUT_PINS:
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
        """Main optimization routine for decoder"""
        print(f"Bayesian optimization for decoder...")

        try:
            # Phase 1: Initial sampling
            self.initial_sampling(n_samples=30)
            
            # Phase 2: Bayesian optimization
            self.bayesian_optimize(n_iterations=15)
            
            # Phase 3: Local exploration around best
            self.explore_around_best(n_samples=8)
            
            # Phase 4: Final Bayesian refinement
            self.bayesian_optimize(n_iterations=10)
            
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
    optimizer = BayesianDecoderOptimizer()
    
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