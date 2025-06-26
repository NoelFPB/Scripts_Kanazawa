import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc
import json
from datetime import datetime
import math
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import random

# Serial port configuration 
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# === DUAL GATE CONFIGURATION ===
GATE_1548 = "AND"   # Logic gate at 1548nm
GATE_1552 = "OR"    # Logic gate at 1552nm
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

# Laser control
LASER_ADDRESS = "GPIB0::6::INSTR"


def generate_truth_table(gate_type):
    """Generate truth table for given gate type"""
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

class DualWavelengthOptimizer:
    """
    Optimize for dual-wavelength logic gate behavior:
    - AND gate at 1548nm
    - OR gate at 1552nm
    Using the same heater configuration
    """
    def __init__(self, gate_1548=GATE_1548, gate_1552=GATE_1552):

        self.gate_1548 = gate_1548
        self.gate_1552 = gate_1552
        self.truth_table_1548 = generate_truth_table(gate_1548)
        self.truth_table_1552 = generate_truth_table(gate_1552)
        
        print(f"Optimizing dual-wavelength logic gates:")
        print(f"  {gate_1548} gate at 1548nm:")
        for inputs, output in self.truth_table_1548.items():
            print(f"    {inputs} -> {'HIGH' if output else 'LOW'}")
        print(f"  {gate_1552} gate at 1552nm:")
        for inputs, output in self.truth_table_1552.items():
            print(f"    {inputs} -> {'HIGH' if output else 'LOW'}")
        
                # Just add these 6 lines to your existing __init__ method:
        self.X_evaluated = []     # Bayesian memory
        self.y_1548 = []         # 1548nm scores  
        self.y_1552 = []         # 1552nm scores
        self.y_combined = []     # Combined scores
        self.gp_1548 = None      # Gaussian Process models
        self.gp_1552 = None
        self.gp_combined = None

        # Initialize hardware connections
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        self.laser = self._init_laser()
        time.sleep(1)
        
        # Best configuration found
        self.best_config = None
        self.best_score = float('-inf')
        self.best_details = {}

        self.base_config = {}
        
        # Set fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                self.base_config[h] = 0.01
                
        # Results tracking
        self.results_log = []

    def config_to_array(self, config):
        """Convert config dict to numpy array"""
        return np.array([config[h] for h in MODIFIABLE_HEATERS])

    def array_to_config(self, x):
        """Convert numpy array to config dict"""
        return {h: x[i] for i, h in enumerate(MODIFIABLE_HEATERS)}

    def add_evaluation(self, config, score_1548, score_1552, score_combined):
        """Add a new evaluation to the Bayesian optimizer dataset"""
        x = self.config_to_array(config)
        self.X_evaluated.append(x)
        self.y_1548.append(score_1548)
        self.y_1552.append(score_1552)
        self.y_combined.append(score_combined)


    def fit_gaussian_processes(self):
        """Robust GP fitting with better hyperparameters"""
        if len(self.X_evaluated) < 5:
            return
        
        X = np.array(self.X_evaluated)
        y = np.array(self.y_combined)
        
        print(f"    Fitting GP with {len(X)} points, y range: [{y.min():.1f}, {y.max():.1f}]")
        
        # Robust kernel with reasonable bounds
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.1, 100)) *
            RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
            WhiteKernel(noise_level=0.1, noise_level_bounds=(0.01, 10.0))
        )
        
        try:
            self.gp_combined = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,  # More restarts for better fit
                alpha=1e-6,
                normalize_y=False,  # Don't normalize with small datasets
                random_state=42
            )
            
            self.gp_combined.fit(X, y)
            
            # Test the fit quality
            y_pred, y_std = self.gp_combined.predict(X[:5], return_std=True)
            mse = np.mean((y_pred - y[:5])**2)
            print(f"    GP fit quality (MSE on first 5 points): {mse:.2f}")
            print(f"    GP fitted successfully!")
            
        except Exception as e:
            print(f"    GP fitting failed: {e}")
            self.gp_combined = None

    def acquisition_function(self, x):
        """Upper Confidence Bound - more robust than EI"""
        if self.gp_combined is None:
            return random.random()
        
        x = x.reshape(1, -1)
        mu, sigma = self.gp_combined.predict(x, return_std=True)
        mu, sigma = mu[0], sigma[0]
        
        # UCB with high exploration
        beta = 5.0  # High exploration parameter
        ucb = mu + beta * sigma
        
        # Add small random component to break ties
        ucb += random.random() * 0.1
        
        return ucb

    def suggest_next_config(self):
        """Proper Bayesian optimization with robust candidate selection"""
        
        if len(self.X_evaluated) < 5:
            # Random until we have enough data
            x = np.random.uniform(0.1, 4.9, len(MODIFIABLE_HEATERS))
            return self.array_to_config(x)
        
        print("  Bayesian candidate selection...")
        
        # Smart candidate generation strategy
        candidates = []
        
        # 1. Random exploration (40%)
        n_random = 2000
        random_candidates = np.random.uniform(0.1, 4.9, size=(n_random, len(MODIFIABLE_HEATERS)))
        candidates.extend(random_candidates)
        
        # 2. Local search around best points (40%)
        best_configs = sorted(zip(self.X_evaluated, self.y_combined), 
                            key=lambda x: x[1], reverse=True)[:5]
        
        n_local = 2000
        for _ in range(n_local):
            # Pick a random good config
            base_x, _ = random.choice(best_configs)
            # Add Gaussian noise
            noise = np.random.normal(0, 0.3, len(MODIFIABLE_HEATERS))
            candidate = np.clip(base_x + noise, 0.1, 4.9)
            candidates.append(candidate)
        
        # 3. Grid-based exploration (20%)
        n_grid = 1000
        for _ in range(n_grid):
            candidate = []
            for _ in range(len(MODIFIABLE_HEATERS)):
                # Discrete values
                val = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
                candidate.append(val)
            candidates.append(np.array(candidate))
        
        # Convert to numpy array
        candidates = np.array(candidates)
        
        # Evaluate acquisition function for all candidates
        print(f"    Evaluating {len(candidates)} candidates...")
        acquisition_values = []
        
        for i, candidate in enumerate(candidates):
            acq_val = self.acquisition_function(candidate)
            acquisition_values.append(acq_val)
        
        # Select best candidate
        best_idx = np.argmax(acquisition_values)
        best_candidate = candidates[best_idx]
        best_acq = acquisition_values[best_idx]
        
        print(f"    Best acquisition value: {best_acq:.4f}")
        
        # Diagnostic info
        acq_array = np.array(acquisition_values)
        if acq_array.std() < 1e-6:
            print(f"    WARNING: All acquisition values identical! Using random fallback.")
            x = np.random.uniform(0.1, 4.9, len(MODIFIABLE_HEATERS))
            return self.array_to_config(x)
        
        print(f"    Acquisition stats: mean={acq_array.mean():.4f}, std={acq_array.std():.4f}")
        
        return self.array_to_config(best_candidate)

    def predict_performance(self, config):
        """Predict performance at a configuration"""
        if self.gp_combined is None:
            return 0, 100  # No prediction possible
        
        x = self.config_to_array(config).reshape(1, -1)
        mu, sigma = self.gp_combined.predict(x, return_std=True)
        return mu[0], sigma[0]
    

    def bayesian_optimize(self, n_iterations=25, initial_random=5, batch_size=5):
        """
        Bayesian optimization with batched evaluations to minimize laser switching
        
        Args:
            n_iterations: Total number of configurations to evaluate
            initial_random: Number of random points before starting GP
            batch_size: Number of configs to evaluate together (reduces laser switching)
        """
        print(f"\nBAYESIAN DUAL-WAVELENGTH OPTIMIZATION")
        print(f"Total iterations: {n_iterations}")
        print(f"Batch size: {batch_size} (reduces laser switching)")
        print("=" * 60)
        
        configs_to_evaluate = []
        iteration = 0
        
        while iteration < n_iterations:
            # Collect batch of configurations
            batch_configs = []
            
            for _ in range(min(batch_size, n_iterations - iteration)):
                if iteration < initial_random:
                    # Random exploration phase
                    print(f"Random exploration iteration {iteration+1}")
                    config = self.array_to_config(
                        np.random.uniform(0.1, 4.9, len(MODIFIABLE_HEATERS))
                    )
                else:
                    # Bayesian optimization phase
                    print(f"Bayesian iteration {iteration+1}")
                    config = self.suggest_next_config()
                    
                    # Show prediction
                    pred_score, uncertainty = self.predict_performance(config)
                    print(f"  Predicted score: {pred_score:.1f} ± {uncertainty:.1f}")
                
                batch_configs.append(config)
                iteration += 1
            
            # Evaluate entire batch (only 2 wavelength switches per batch!)
            print(f"\nEvaluating batch of {len(batch_configs)} configurations...")
            results = self.evaluate_configs_dual_wavelength(batch_configs)
            
            # Process results and update Bayesian optimizer
            for i, (config, combined_score, details) in enumerate(results):
                # Extract individual wavelength scores
                score_1548 = details['1548nm'][0] if '1548nm' in details else -1000
                score_1552 = details['1552nm'][0] if '1552nm' in details else -1000
                
                # Add to Bayesian optimizer
                self.add_evaluation(config, score_1548, score_1552, combined_score)
                
                print(f"  Config {i+1}: 1548nm={score_1548:.1f}, 1552nm={score_1552:.1f}, Combined={combined_score:.1f}")
            
            # Fit GP models after each batch
            if len(self.X_evaluated) >= 3:
                print("Updating Gaussian Process models...")
                self.fit_gaussian_processes()
        
        print(f"\nBAYESIAN OPTIMIZATION COMPLETE")
        print(f"Best combined score: {self.best_score:.1f}")
        print(f"Total evaluations: {len(self.X_evaluated)}")

    def _init_scope(self):
        """Initialize oscilloscope"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No scope VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
        
        # Setup Channel 2 for logic gate output measurement
        scope.write(':CHANnel2:DISPlay ON')
        scope.write(':CHANnel2:SCALe 2')
        scope.write(':CHANnel2:OFFSet -6')
        
        return scope
    
    def _init_laser(self):
        """Initialize laser for wavelength control"""
        rm = pyvisa.ResourceManager()
        laser = rm.open_resource(LASER_ADDRESS)
        laser.timeout = 5000
        laser.write_termination = ''
        laser.read_termination = ''
        print("Connected to laser")
        return laser
    
    def set_wavelength(self, wavelength_nm):
        """Set laser wavelength"""
        try:
            self.laser.write(f'LW{wavelength_nm}nm')
            time.sleep(2)  # Wait for wavelength to settle
            # Verify wavelength
            try:
                status = self.laser.query('LS?')
                print(f"  Wavelength set: {status}")
            except:
                print(f"  Wavelength set to {wavelength_nm}nm")
            return True
        except Exception as e:
            print(f"Wavelength setting failed: {e}")
            return False
    
    def turn_laser_on(self):
        """Turn laser ON"""
        try:
            self.laser.write('LE1')
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Laser ON failed: {e}")
            return False
    
    def turn_laser_off(self):
        """Turn laser OFF"""
        try:
            self.laser.write('LE0')
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Laser OFF failed: {e}")
            return False
    
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

    def calculate_balanced_score(self, high_outputs, low_outputs, gate_name):
        """Simple, realistic scoring for photonic logic gates
        Focuses on the essentials: clear separation, reliability, and practical thresholds"""
        
        if not high_outputs or not low_outputs:
            return -50
        
        min_high = min(high_outputs)
        max_low = max(low_outputs)
        mean_high = sum(high_outputs) / len(high_outputs)
        mean_low = sum(low_outputs) / len(low_outputs)
        
        # HARD REQUIREMENT: Logic levels must not overlap
        if min_high <= max_low:
            overlap = max_low - min_high
            return -100 - (overlap * 20)  # Severe penalty scaling with overlap
        
        # 1. SEPARATION QUALITY (60 points) - Most important
        # The gap between worst HIGH and worst LOW
        separation = min_high - max_low
        
        # Realistic thresholds for photonic devices:
        # 0.3V = barely working, 0.8V = good, 1.5V+ = excellent
        if separation < 0.3:
            separation_score = 0
        elif separation < 0.8:
            separation_score = 30 * (separation - 0.3) / 0.5  # Linear 0→30
        else:
            separation_score = 30 + 30 * (1 - math.exp(-(separation - 0.8)))  # Saturating 30→60
        
        # 2. SIGNAL QUALITY (25 points)
        # HIGH states should be reasonably high, LOW states reasonably low
        signal_ratio = mean_high / max(mean_low, 0.05)  # Avoid division by zero
        
        # Realistic ratios: 3x = minimal, 10x = good, 50x+ = excellent
        if signal_ratio < 3:
            signal_score = 0
        elif signal_ratio < 10:
            signal_score = 15 * (signal_ratio - 3) / 7  # Linear 0→15
        else:
            signal_score = 15 + 10 * (1 - math.exp(-(signal_ratio - 10) / 20))  # Saturating 15→25
        
        # 3. REPEATABILITY (15 points)
        # States should be consistent - simple range check
        high_range = max(high_outputs) - min(high_outputs)
        low_range = max(low_outputs) - min(low_outputs)
        
        # Accept up to 15% variation within each state
        high_variation = high_range / max(mean_high, 0.1)
        low_variation = low_range / max(mean_low, 0.1)
        avg_variation = (high_variation + low_variation) / 2
        
        if avg_variation < 0.15:  # Less than 15% variation
            repeatability_score = 15
        elif avg_variation < 0.30:  # 15-30% variation
            repeatability_score = 15 * (0.30 - avg_variation) / 0.15
        else:
            repeatability_score = 0  # Too much variation
        
        total_score = separation_score + signal_score + repeatability_score
        return min(100, max(0, total_score))

    # def calculate_balanced_score(self, high_outputs, low_outputs, gate_name):
    #     """Scoring optimized for extinction ratio as calculated in the paper
    #     Maximizes the ratio of largest output to second-largest output"""
        
    #     if not high_outputs or not low_outputs:
    #         return -50
        
    #     # Combine all outputs and sort to find largest and second-largest
    #     all_outputs = high_outputs + low_outputs
    #     all_outputs.sort(reverse=True)  # Sort in descending order
        
    #     largest_output = all_outputs[0]
    #     second_largest_output = all_outputs[1]
        
    #     # Basic overlap check using traditional method
    #     min_high = min(high_outputs)
    #     max_low = max(low_outputs)
        
    #     # HARD REQUIREMENT: Logic levels must not overlap
    #     if min_high <= max_low:
    #         overlap = max_low - min_high
    #         return -100 - (overlap * 20)
        
    #     # 1. EXTINCTION RATIO (50 points) - Primary metric from paper
    #     # ER = 10 * log10(largest / second_largest)
    #     er_linear = largest_output / max(second_largest_output, 0.001)
    #     er_db = 10 * math.log10(er_linear)
        
    #     # Paper shows ~4-5 dB is good, 7+ dB is excellent
    #     # Map extinction ratio to score: 3dB→10pts, 5dB→30pts, 7dB→45pts, 10dB→50pts
    #     if er_db < 1:
    #         er_score = 0
    #     elif er_db < 2:
    #         er_score = 15 * (er_db - 3) / 2  # Linear 0→15
    #     elif er_db < 5:
    #         er_score = 15 + 20 * (er_db - 5) / 2  # Linear 15→35
    #     else:
    #         er_score = 35 + 15 * (1 - math.exp(-(er_db - 7) / 3))  # Saturating 35→50
        
    #     # 2. CLEAR WINNER (30 points) - Ensure one output is clearly dominant
    #     # The largest should be significantly larger than the second-largest
    #     dominance_ratio = largest_output / max(second_largest_output, 0.001)
        
    #     # Linear ratio thresholds: 2x = minimal, 5x = good, 10x+ = excellent
    #     if dominance_ratio < 2:
    #         dominance_score = 0
    #     elif dominance_ratio < 5:
    #         dominance_score = 15 * (dominance_ratio - 2) / 3  # Linear 0→15
    #     else:
    #         dominance_score = 15 + 15 * (1 - math.exp(-(dominance_ratio - 5) / 5))  # Saturating 15→30
        
    #     # 3. OUTPUT LEVEL QUALITY (20 points) - Ensure reasonable operating levels
    #     mean_high = sum(high_outputs) / len(high_outputs)
    #     mean_low = sum(low_outputs) / len(low_outputs)
        
    #     # HIGH should be reasonably high (>1V ideal), LOW should be reasonably low (<1V ideal)
    #     high_quality = min(1.0, largest_output / 2.0)  # Normalize to 2V reference
    #     low_quality = max(0.0, 1.0 - min(all_outputs) / 1.0)  # Penalize if minimum > 1V
    #     level_score = 20 * (high_quality + low_quality) / 2
        
    #     total_score = er_score + dominance_score + level_score
    #     return min(100, max(0, total_score))
     

    def evaluate_single_wavelength_batch(self, configs, wavelength, truth_table, gate_name):
        """Evaluate multiple configurations at a single wavelength (batch mode)"""
        print(f"\nSetting laser to {wavelength}nm for {gate_name} gate testing...")
        
        # Set wavelength once
        if not self.set_wavelength(wavelength):
            return [(config, -1000, {}) for config in configs]
        
        if not self.turn_laser_on():
            return [(config, -1000, {}) for config in configs]
        
        print(f"Waiting for laser to stabilize...")
        time.sleep(14)  # Wait for laser to reach operating condition
        
        print(f"Laser stable at {wavelength}nm. Testing {len(configs)} configurations...")
        
        results = []
        
        for i, config in enumerate(configs):
            print(f"  Testing config {i+1}/{len(configs)}...")
            score, details = self._evaluate_single_config_at_current_wavelength(
                config, truth_table, gate_name, wavelength)
            results.append((config, score, details))
            
            if score > 50:  # Only show promising results
                print(f"    Score: {score:.1f} (promising)")
            else:
                print(f"    Score: {score:.1f}")
        
        self.turn_laser_off()
        return results
    
    def _evaluate_single_config_at_current_wavelength(self, config, truth_table, gate_name, wavelength):
        """Evaluate a single configuration at the current wavelength"""
        
        # Collect outputs
        high_outputs = []
        low_outputs = []
        detailed_results = {}
        
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            expected_high = truth_table[input_state]
            
            self.send_heater_values(current_config)
            time.sleep(0.25)  # Just heater settling time, not laser
            
            output = self.measure_output()
            if output is None:
                return -1000, {}
            
            detailed_results[input_state] = {
                'output': output,
                'expected_high': expected_high
            }
            
            if expected_high:
                high_outputs.append(output)
            else:
                low_outputs.append(output)
        
        # If missing states, penalize
        if not high_outputs or not low_outputs:
            return -500, detailed_results
        
           # USE THE SINGLE BALANCED SCORING
        final_score = self.calculate_balanced_score(high_outputs, low_outputs, gate_name)
        
        return final_score, detailed_results

    def evaluate_configs_dual_wavelength(self, configs):
        '''Here we determine the combined score, that is an evaluation of how work this wells in regards of the 2 logic gates'''

        print(f"\nDUAL-WAVELENGTH BATCH EVALUATION")
        print(f"Evaluating {len(configs)} configurations at both wavelengths...")
        
        # Batch evaluate at 1548nm
        results_1548 = self.evaluate_single_wavelength_batch(
            configs, 1548, self.truth_table_1548, self.gate_1548)
        
        time.sleep(1)  # Brief pause between wavelengths
        
        # Batch evaluate at 1552nm
        results_1552 = self.evaluate_single_wavelength_batch(
            configs, 1552, self.truth_table_1552, self.gate_1552)
        
        # Combine results
        combined_results = []
        for i, config in enumerate(configs):
            _, score_1548, details_1548 = results_1548[i]
            _, score_1552, details_1552 = results_1552[i]
            
            # Combined scoring - MUCH more forgiving
            if score_1548 < -500 or score_1552 < -500:
                combined_score = -2000
            else:
                # Reward ANY positive performance
                positive_1548 = max(0, score_1548)
                positive_1552 = max(0, score_1552)
                
                if positive_1548 > 0 and positive_1552 > 0:
                    # Both positive: excellent!
                    combined_score = (positive_1548 + positive_1552) * 1.5
                elif positive_1548 > 0 or positive_1552 > 0:
                    # One positive: still valuable for learning
                    single_good = max(positive_1548, positive_1552)
                    combined_score = single_good * 0.6  # Reduced but positive
                else:
                    # Both negative: take the less bad one
                    combined_score = max(score_1548, score_1552) * 0.3    
                # Update best if this is better

            if combined_score > self.best_score:
                self.best_score = combined_score
                self.best_config = config.copy()
                self.best_details = {
                    '1548nm_AND': details_1548,
                    '1552nm_OR': details_1552,
                    'scores': {
                        '1548nm': score_1548,
                        '1552nm': score_1552,
                        'combined': combined_score
                    }
                }
                
                print(f"  NEW BEST dual-gate configuration!")
                print(f"     1548nm {self.gate_1548}: {score_1548:.1f}")
                print(f"     1552nm {self.gate_1552}: {score_1552:.1f}")
                print(f"     Combined: {combined_score:.1f}")
            
            combined_results.append((config, combined_score, {
                '1548nm': (score_1548, details_1548),
                '1552nm': (score_1552, details_1552)
            }))
        
        return combined_results
    
    def initial_sampling(self, n_samples=20):
        """Initial sampling using batch evaluation at both wavelengths"""
        print(f"INITIAL SAMPLING - Batch mode for {n_samples} configurations")
        
        # Generate all configurations first
        configs = []
        
        # Zero configuration
        configs.append({h: 0.1 for h in MODIFIABLE_HEATERS})
        
        # Latin Hypercube Sampling for the rest
        n_dims = len(MODIFIABLE_HEATERS)
        sampler = qmc.LatinHypercube(d=n_dims, seed=42)
        samples = sampler.random(n=n_samples-1)
        
        for sample in samples:
            config = {}
            for j, h in enumerate(MODIFIABLE_HEATERS):
                config[h] = V_MIN + sample[j] * (V_MAX - V_MIN)
            configs.append(config)
        
        # Evaluate all configurations in batch mode
        results = self.evaluate_configs_dual_wavelength(configs)
        
        print(f"Initial sampling complete. Best combined score: {self.best_score:.1f}")
        
        return results
    
    def spsa_batch_optimize(self, iterations=10, batch_size=6, a=0.8, c=0.3, alpha=0.35, gamma=0.101):
        """Real SPSA optimization using batch evaluation"""
        
        if not self.best_config:
            print("No initial configuration available for SPSA")
            return
        
        theta = {h: self.best_config.get(h, 0.1) for h in MODIFIABLE_HEATERS}
        heater_keys = sorted(MODIFIABLE_HEATERS)
        
        print(f"\nSPSA BATCH OPTIMIZATION")
        print(f"Iterations: {iterations}, Batch size: {batch_size}")
        
        for k in range(1, iterations + 1):
            print(f"\n--- SPSA Iteration {k}/{iterations} ---")
            
            # Update parameters
            ak = a / (k ** alpha)
            ck = max(0.3, c / (k ** gamma))
            
            # Generate batch of configurations to test
            configs_to_test = []
            
            for b in range(batch_size):
                # Generate perturbation
                delta = {h: 1 if random.random() > 0.5 else -1 for h in heater_keys}
                
                # Create perturbed configurations
                theta_plus = {h: max(0.1, min(4.9, theta[h] + ck * delta[h])) for h in heater_keys}
                theta_minus = {h: max(0.1, min(4.9, theta[h] - ck * delta[h])) for h in heater_keys}
                
                configs_to_test.append((theta_plus, theta_minus, delta))
            
            # Flatten configurations for batch evaluation
            all_configs = []
            for theta_plus, theta_minus, _ in configs_to_test:
                all_configs.extend([theta_plus, theta_minus])
            
            print(f"Evaluating {len(all_configs)} configurations in batch...")
            results = self.evaluate_configs_dual_wavelength(all_configs)

            # Calculate average gradient across all perturbations
            total_gradient = {h: 0.0 for h in heater_keys}
            valid_gradients = 0
            
            for i, (theta_plus, theta_minus, delta) in enumerate(configs_to_test):
                # Get scores for this perturbation pair
                idx_plus = i * 2
                idx_minus = i * 2 + 1
                
                # Fix: results contains (config, combined_score, details)
                _, score_plus, _ = results[idx_plus]  # score_plus is the combined_score (float)
                _, score_minus, _ = results[idx_minus]  # score_minus is the combined_score (float)

                # Skip if either evaluation failed
                if score_plus < -500 or score_minus < -500:
                    continue
                
                # Calculate gradient estimate for this perturbation
                for h in heater_keys:
                    gradient_component = (score_plus - score_minus) / (2 * ck * delta[h])
                    total_gradient[h] += gradient_component
                valid_gradients += 1
            
            # Update theta using average gradient (THIS IS SPSA!)
            if valid_gradients > 0:
                for h in heater_keys:
                    avg_gradient = total_gradient[h] / valid_gradients
                    theta[h] = max(0.1, min(4.9, theta[h] + ak * avg_gradient))
                     
            else:
                print("No valid gradients, keeping current theta")
            
            print(f"Step size: {ak:.3f}, Perturbation: {ck:.3f}, Best: {self.best_score:.1f}")

    def focused_search_around_best(self, n_variations=12):
        """Generate variations around the best configuration and test in batch"""
        if not self.best_config:
            print("No best configuration for focused search")
            return
        
        print(f"\nFOCUSED SEARCH around best configuration")
        print(f"Generating {n_variations} variations...")
        
        base_config = self.best_config.copy()
        variations = []
        
        # Generate systematic variations
        for i in range(n_variations):
            variation = base_config.copy()
            
            # Randomly select heaters to modify
            heaters_to_modify = random.sample(MODIFIABLE_HEATERS, 
                                            random.randint(1, min(5, len(MODIFIABLE_HEATERS))))
            
            for h in heaters_to_modify:
                current = variation[h]
                # Small focused perturbations
                perturbation = random.uniform(-0.2, 0.2)
                variation[h] = max(0.1, min(4.9, current + perturbation))
            
            variations.append(variation)
        
        # Test all variations in batch
        results = self.evaluate_configs_dual_wavelength(variations)
        
        print(f"Focused search complete. Best score: {self.best_score:.1f}")
        
        return results
    
    def test_final_configuration(self):
        """Test the final configuration at both wavelengths"""
        if not self.best_config:
            print("No configuration to test.")
            return
        
        print(f"\nTESTING FINAL DUAL-WAVELENGTH CONFIGURATION")
        print("=" * 60)

        # Test at 1548nm (AND gate)
        print(f"\nTesting at 1548nm ({self.gate_1548} gate):")
        self.set_wavelength(1548)
        self.turn_laser_on()
        time.sleep(14)
        
        outputs_1548 = []
        high_outputs_1548 = []
        low_outputs_1548 = []
        
        for input_state in INPUT_COMBINATIONS:
            current_config = self.best_config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.3)
            output = self.measure_output()
            expected = self.truth_table_1548[input_state]
            
            outputs_1548.append(output)
            if expected:
                high_outputs_1548.append(output)
            else:
                low_outputs_1548.append(output)
            
            print(f"  Inputs {input_state}: {output:.4f}V (expect {'HIGH' if expected else 'LOW'})")
        
        # Calculate and print extinction ratio for 1548nm (using paper's method)
        if high_outputs_1548 and low_outputs_1548:
            # Paper's method: ratio of largest to second-largest output
            all_outputs_1548 = outputs_1548[:]
            all_outputs_1548.sort(reverse=True)  # Sort in descending order
            largest_output = all_outputs_1548[0]
            second_largest_output = all_outputs_1548[1]
            
            er_linear_1548_paper = largest_output / max(second_largest_output, 0.001)
            er_db_1548_paper = 10 * math.log10(er_linear_1548_paper)
            
            # Also calculate traditional method for comparison
            mean_high_1548 = sum(high_outputs_1548) / len(high_outputs_1548)
            mean_low_1548 = sum(low_outputs_1548) / len(low_outputs_1548)
            er_linear_1548_trad = mean_high_1548 / max(mean_low_1548, 0.001)
            er_db_1548_trad = 10 * math.log10(er_linear_1548_trad)
            
            separation_1548 = min(high_outputs_1548) - max(low_outputs_1548)
            
            print(f"  --> Extinction Ratio (paper method): {er_db_1548_paper:.2f} dB (linear: {er_linear_1548_paper:.1f})")
            print(f"  --> Extinction Ratio (traditional): {er_db_1548_trad:.2f} dB (linear: {er_linear_1548_trad:.1f})")
            print(f"  --> Worst-case separation: {separation_1548:.3f}V")
            print(f"  --> HIGH levels: {mean_high_1548:.3f}V ± {(max(high_outputs_1548) - min(high_outputs_1548))/2:.3f}V")
            print(f"  --> LOW levels:  {mean_low_1548:.3f}V ± {(max(low_outputs_1548) - min(low_outputs_1548))/2:.3f}V")
        
        self.turn_laser_off()

        # Test at 1552nm (OR gate)
        print(f"\nTesting at 1552nm ({self.gate_1552} gate):")
        self.set_wavelength(1552)
        self.turn_laser_on()
        time.sleep(14)
        
        outputs_1552 = []
        high_outputs_1552 = []
        low_outputs_1552 = []
        
        for input_state in INPUT_COMBINATIONS:
            current_config = self.best_config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.3)
            output = self.measure_output()
            expected = self.truth_table_1552[input_state]
            
            outputs_1552.append(output)
            if expected:
                high_outputs_1552.append(output)
            else:
                low_outputs_1552.append(output)
            
            print(f"  Inputs {input_state}: {output:.4f}V (expect {'HIGH' if expected else 'LOW'})")
        
        # Calculate and print extinction ratio for 1552nm (using paper's method)
        if high_outputs_1552 and low_outputs_1552:
            # Paper's method: ratio of largest to second-largest output
            all_outputs_1552 = outputs_1552[:]
            all_outputs_1552.sort(reverse=True)  # Sort in descending order
            largest_output = all_outputs_1552[0]
            second_largest_output = all_outputs_1552[1]
            
            er_linear_1552_paper = largest_output / max(second_largest_output, 0.001)
            er_db_1552_paper = 10 * math.log10(er_linear_1552_paper)
            
            # Also calculate traditional method for comparison
            mean_high_1552 = sum(high_outputs_1552) / len(high_outputs_1552)
            mean_low_1552 = sum(low_outputs_1552) / len(low_outputs_1552)
            er_linear_1552_trad = mean_high_1552 / max(mean_low_1552, 0.001)
            er_db_1552_trad = 10 * math.log10(er_linear_1552_trad)
            
            separation_1552 = min(high_outputs_1552) - max(low_outputs_1552)
            
            print(f"  --> Extinction Ratio (paper method): {er_db_1552_paper:.2f} dB (linear: {er_linear_1552_paper:.1f})")
            print(f"  --> Extinction Ratio (traditional): {er_db_1552_trad:.2f} dB (linear: {er_linear_1552_trad:.1f})")
            print(f"  --> Worst-case separation: {separation_1552:.3f}V")
            print(f"  --> HIGH levels: {mean_high_1552:.3f}V ± {(max(high_outputs_1552) - min(high_outputs_1552))/2:.3f}V")
            print(f"  --> LOW levels:  {mean_low_1552:.3f}V ± {(max(low_outputs_1552) - min(low_outputs_1552))/2:.3f}V")
        
        self.turn_laser_off()

    
    def format_config(self):
        """Format the final configuration"""
        if not self.best_config:
            return {}
        
        complete_config = {i: 0.0 for i in range(40)}
        
        for heater, value in self.best_config.items():
            complete_config[heater] = value
            
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                complete_config[h] = 0.01
                
        return {k: round(float(v), 3) for k, v in complete_config.items()}
    
    def save_results(self):
        """Save results to file with proper JSON serialization"""
        if not self.best_config:
            return
        
        # Convert any tuples to strings for JSON compatibility
        def serialize_for_json(obj):
            if isinstance(obj, dict):
                return {str(k): serialize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_for_json(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'gate_1548': self.gate_1548,
            'gate_1552': self.gate_1552,
            'best_score': float(self.best_score),
            'best_config': self.format_config(),
            'best_details': serialize_for_json(self.best_details) if self.best_details else {}
        }
        
        filename = f"dual_wavelength_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
            # At least print the configuration so it's not lost
            print("Final configuration (copy this):")
            print(json.dumps(self.format_config(), indent=2))
    
    def cleanup(self):
        """Clean up connections safely"""
        print("Cleaning up connections...")
        
        # Turn off laser safely
        try:
            if hasattr(self, 'laser') and self.laser:
                self.laser.write('LE0')
                time.sleep(0.5)
                print("Laser turned OFF")
        except Exception as e:
            print(f"Warning: Could not turn off laser: {e}")
        
        # Close laser connection
        try:
            if hasattr(self, 'laser') and self.laser:
                self.laser.close()
                print("Laser connection closed")
        except Exception as e:
            print(f"Warning: Could not close laser connection: {e}")
        
        # Close serial connection
        try:
            if hasattr(self, 'serial') and self.serial:
                self.serial.close()
                print("Serial connection closed")
        except Exception as e:
            print(f"Warning: Could not close serial connection: {e}")
        
        # Close scope connection
        try:
            if hasattr(self, 'scope') and self.scope:
                self.scope.close()
                print("Scope connection closed")
        except Exception as e:
            print(f"Warning: Could not close scope connection: {e}")
        
        print("Cleanup complete")
    
    def optimize(self):
        print(f"\nDUAL-WAVELENGTH BATCH OPTIMIZATION STARTING")
        print(f"Target: {self.gate_1548} at 1548nm, {self.gate_1552} at 1552nm")
        print(f"Laser settling time: 25 seconds per wavelength")
        print("=" * 60)
        
        try:
            # Phase 1: Initial sampling (batch mode)
            print(f"\nPHASE 1: Initial Sampling")
            self.initial_sampling(n_samples=30)

            print(f"\nPHASE 2: Bayesian Optimization")
            self.bayesian_optimize(n_iterations=30, initial_random=0, batch_size=10)
            
            # Phase 2: SPSA optimization (batch mode)
            #print(f"\nPHASE 2: SPSA Optimization")
            #self.spsa_batch_optimize(iterations=10, batch_size=2) #batch is how many perturbations I do to estimate the gradient
            # But I am doing batching because before every iteration I determine a good gradient for both gates
            # So  I need to switch between the wavelenghts for each iteration, thats why batching makes sense
            
            # Phase 3: Focused search around best
            #print(f"\nPHASE 3: Focused Search")
            #self.focused_search_around_best(n_variations=10)
            
            # Phase 4: Final testing
            print(f"\nPHASE 4: Final Validation")
            self.test_final_configuration()
            
            print(f"\nFINAL CONFIGURATION:")
            config = self.format_config()
            print(config)
            
            self.save_results()
            
            return self.best_config, self.best_score
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None, -1
        
        finally:
            self.cleanup()

def main():
    start_time = time.time()
    optimizer = DualWavelengthOptimizer(GATE_1548, GATE_1552)
    
    try:
        best_config, best_score = optimizer.optimize()
        
        if best_config:
            print(f"\nOPTIMIZATION COMPLETE!")
            print(f"Best combined score: {best_score:.2f}")
        else:
            print(f"\nOPTIMIZATION FAILED")
            
    except KeyboardInterrupt:
        print(f"\nOptimization interrupted by user")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    # actually I am already calling cleanup before
    finally:
        try:
            optimizer.cleanup()
        except:
            pass
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time/60:.1f} minutes")

if __name__ == "__main__":
    main()