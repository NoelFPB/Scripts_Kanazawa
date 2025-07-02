import serial
import time
import pyvisa
import numpy as np
import random
import pickle
import os
from datetime import datetime
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Serial port configuration 
SERIAL_PORT = 'COM3'
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

    def __init__(self):
        print("Initializing Bayesian decoder optimization...")
        
        # Simple save file
        self.save_file = "decoder_model.pkl"
        
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
        self.total_evaluations = 0

        self.base_config = {}
        
        # Set fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_PINS:
                self.base_config[h] = 0.01

        # Try to load existing model
        self.load_model()

    def save_model(self):
        """Save the complete model state"""
        state = {
            'X_evaluated': self.X_evaluated,
            'y_evaluated': self.y_evaluated,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'total_evaluations': self.total_evaluations,
            # Don't save GP model to avoid version conflicts
            'gp': None  
        }
        
        with open(self.save_file, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved with {len(self.X_evaluated)} data points, best score: {self.best_score:.2f}")

    def load_model(self):
        """Load existing model if available"""
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, 'rb') as f:
                    state = pickle.load(f)
                
                self.X_evaluated = state['X_evaluated']
                self.y_evaluated = state['y_evaluated']
                self.best_config = state['best_config']
                self.best_score = state['best_score']
                self.total_evaluations = state.get('total_evaluations', len(self.y_evaluated))
                # Always retrain GP from data (don't load saved GP)
                self.gp = None
                
                print(f"Loaded existing model: {len(self.X_evaluated)} data points, best score: {self.best_score:.2f}")
                print("Will retrain GP from loaded data...")
                
                # Retrain GP immediately if we have enough data
                if len(self.X_evaluated) >= 3:
                    self.fit_gaussian_process()
                
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting fresh...")
        else:
            print("No existing model found. Starting fresh...")

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
        Hybrid decoder scoring: Focus on extinction ratio but provide gradient guidance
        for complex decoder optimization landscape
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
        
        # Basic statistics
        min_active = min(all_active_outputs)
        max_inactive = max(all_inactive_outputs)
        mean_active = np.mean(all_active_outputs)
        mean_inactive = np.mean(all_inactive_outputs)
        separation = min_active - max_inactive
        
        # === PRIMARY METRIC: EXTINCTION RATIO (50 points) ===
        if separation > 0:
            # Working decoder - extinction ratio scoring
            er_linear = min_active / max(max_inactive, 0.001)
            er_db = 10 * np.log10(er_linear)
            
            if er_db < 1:
                er_score = 25 * er_db  # 0→25 for 0-1dB
            elif er_db < 3:
                er_score = 25 + 15 * (er_db - 1) / 2  # 25→40 for 1-3dB
            elif er_db < 5:
                er_score = 40 + 8 * (er_db - 3) / 2   # 40→48 for 3-5dB
            else:
                er_score = 48 + 2 * (1 - np.exp(-(er_db - 5) / 3))  # 48→50 for 5+dB
        else:
            # No separation - but still provide gradient based on gap size
            er_score = -20 + 15 * max(0, (separation + 2.0) / 2.0)  # -20 to -5 as gap closes
        
        # === SEPARATION GRADIENT (25 points) ===
        # Always reward moving toward separation, even if not there yet
        mean_separation = mean_active - mean_inactive
        
        if mean_separation > 0:
            # Moving in right direction
            sep_score = 20 * min(1.0, mean_separation / 2.0)  # 0→20 for 0-2V mean separation
            
            # Bonus for actual worst-case separation
            if separation > 0:
                sep_score += 5 * min(1.0, separation / 1.0)  # +0→5 for 0-1V worst-case separation
        else:
            # Wrong direction but still provide gradient
            sep_score = -10 + 10 * max(0, (mean_separation + 2.0) / 2.0)  # -10 to 0 as approaches zero
        
        # === SIGNAL STRENGTH (15 points) ===
        # Reward higher active outputs for better SNR
        strength_score = 15 * min(1.0, mean_active / 4.0)
        
        # === CONSISTENCY (10 points) ===
        # Reward consistent outputs within each group
        active_std = np.std(all_active_outputs)
        inactive_std = np.std(all_inactive_outputs)
        avg_std = (active_std + inactive_std) / 2
        consistency_score = 10 * np.exp(-avg_std * 3)  # Slightly more forgiving than before
        
        # === DIRECTIONALITY BONUS (EXTRA GRADIENT) ===
        # Reward configurations where most actives > most inactives (even if not perfect)
        correct_pairs = 0
        total_pairs = 0
        
        for active_val in all_active_outputs:
            for inactive_val in all_inactive_outputs:
                total_pairs += 1
                if active_val > inactive_val:
                    correct_pairs += 1
        
        if total_pairs > 0:
            directionality = correct_pairs / total_pairs
            direction_bonus = 10 * directionality  # 0→10 points for getting direction right
        else:
            direction_bonus = 0
        
        # === COMBINE SCORES ===
        total_score = er_score + sep_score + strength_score + consistency_score + direction_bonus
        
        # Less aggressive penalties than before (keep gradient)
        if len(all_active_outputs) > 1:
            active_range = max(all_active_outputs) - min(all_active_outputs)
            active_penalty = -10 * min(1.0, active_range / 2.0)  # Less harsh penalty
            total_score += active_penalty

        if len(all_inactive_outputs) > 1:
            inactive_range = max(all_inactive_outputs) - min(all_inactive_outputs)
            inactive_penalty = -5 * min(1.0, inactive_range / 2.0)  # Less harsh penalty
            total_score += inactive_penalty
        
        # Wider range to maintain gradient
        final_score = min(100, max(-30, total_score))
        
        # Enhanced debug output
        if final_score > self.best_score:
            print(f"  === NEW BEST DECODER CONFIGURATION ===")
            if separation > 0:
                er_linear = min_active / max_inactive
                er_db = 10 * np.log10(er_linear)
                print(f"  ✅ WORKING DECODER!")
                print(f"  Extinction Ratio: {er_db:.2f}dB (linear: {er_linear:.1f})")
            else:
                print(f"  🔧 IMPROVING (not working yet)")
                print(f"  Worst-case gap: {separation:.3f}V")
            
            print(f"  Mean separation: {mean_separation:.3f}V")
            print(f"  ACTIVE: {[f'{h:.3f}V' for h in all_active_outputs]} (avg: {mean_active:.3f}V)")
            print(f"  INACTIVE: {[f'{l:.3f}V' for l in all_inactive_outputs[:6]]}... (avg: {mean_inactive:.3f}V)")
            print(f"  Directionality: {directionality:.1%} correct pairs")
            print(f"  Score breakdown: ER={er_score:.1f}, Sep={sep_score:.1f}, Str={strength_score:.1f}, Cons={consistency_score:.1f}, Dir={direction_bonus:.1f}")
            print(f"  Final Score: {final_score:.2f}")
        
        return final_score


    def evaluate_configuration_non_simplified(self, config):       
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
            time.sleep(0.20)  # Increased settling time for thermal stability
            
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
        
        # === ADAPTIVE ANALYSIS: UNDERSTAND NATURAL SYSTEM BEHAVIOR ===
        all_outputs = all_active_outputs + all_inactive_outputs
        
        # Natural statistics of the system
        overall_mean = np.mean(all_outputs)
        overall_std = np.std(all_outputs)
        overall_min = min(all_outputs)
        overall_max = max(all_outputs)
        overall_range = overall_max - overall_min
        
        # Statistics for each group
        active_mean = np.mean(all_active_outputs)
        inactive_mean = np.mean(all_inactive_outputs)
        active_std = np.std(all_active_outputs)
        inactive_std = np.std(all_inactive_outputs)
        
        # === METRIC 1: NATURAL SEPARATION (40 points) ===
        # Reward configurations that naturally separate the two groups
        min_active = min(all_active_outputs)
        max_inactive = max(all_inactive_outputs)  
        separation = min_active - max_inactive
        
        # Also consider mean separation for robustness
        mean_separation = active_mean - inactive_mean
        
        separation_score = 0
        if separation > 0:
            # Working separation - reward based on multiple factors
            
            # Base separation reward (0-25 points)
            normalized_separation = separation / max(overall_range, 0.1)
            separation_score += 25 * min(1.0, normalized_separation * 2)
            
            # Mean separation bonus (0-10 points) 
            if mean_separation > 0:
                normalized_mean_sep = mean_separation / max(overall_range, 0.1)
                separation_score += 10 * min(1.0, normalized_mean_sep)
            
            # Extinction ratio bonus (0-5 points)
            if max_inactive > 0.001:
                er_linear = min_active / max_inactive
                if er_linear > 1.1:  # At least 10% better
                    er_bonus = 5 * min(1.0, (er_linear - 1.0) / 2.0)
                    separation_score += er_bonus
        
        else:
            # Overlapping groups - penalty based on overlap severity
            overlap = abs(separation)
            overlap_penalty = min(20, 20 * (overlap / max(overall_range, 0.1)))
            separation_score = -overlap_penalty



        

        # === METRIC 2: POLARIZATION REWARD (30 points) ===
        # Reward configurations that push outputs toward natural extremes
        polarization_score = 0
        
        # Reward active outputs being high relative to overall distribution
        for active_val in all_active_outputs:
            if overall_range > 0:
                position = (active_val - overall_min) / overall_range
                if position > 0.5:  # Above median
                    polarization_score += 3 * (position - 0.5) * 2  # Up to 3 points each
        
        # Reward inactive outputs being low relative to overall distribution  
        for inactive_val in all_inactive_outputs:
            if overall_range > 0:
                position = (inactive_val - overall_min) / overall_range
                if position < 0.5:  # Below median
                    polarization_score += 2 * (0.5 - position) * 2  # Up to 2 points each
        
        # Cap polarization score
        polarization_score = min(30, polarization_score)
        
        # === METRIC 3: GROUP CONSISTENCY (20 points) ===
        # Reward tight grouping within each logic state
        consistency_score = 0
        
        # Active group consistency (0-10 points)
        if overall_range > 0:
            active_relative_std = active_std / overall_range
            consistency_score += 10 * np.exp(-active_relative_std * 10)
        
        # Inactive group consistency (0-10 points)  
        if overall_range > 0:
            inactive_relative_std = inactive_std / overall_range
            consistency_score += 10 * np.exp(-inactive_relative_std * 10)
        
        # === METRIC 4: BIMODAL DISTRIBUTION REWARD (10 points) ===
        # Reward configurations that create two distinct peaks
        bimodal_score = 0
        
        # Check if the two groups are well-separated in the distribution
        if len(all_outputs) >= 8:  # Need enough samples
            # Simple bimodality test: gap between groups vs within-group spread
            group_gap = abs(active_mean - inactive_mean)
            combined_spread = (active_std + inactive_std) / 2
            
            if combined_spread > 0:
                bimodal_ratio = group_gap / combined_spread
                bimodal_score = 10 * min(1.0, bimodal_ratio / 3.0)  # Full points for 3:1 ratio
        
        # === COMBINE SCORES ===
        total_score = separation_score + polarization_score + consistency_score + bimodal_score
        
        # Apply gentle bounds (wider range for more exploration)
        final_score = min(100, max(-50, total_score))
        
        # Update counters
        self.total_evaluations += 1
        
        # Auto-save every 10 evaluations
        if self.total_evaluations % 10 == 0:
            self.save_model()
        
        # === DETAILED DEBUG OUTPUT FOR IMPROVEMENTS ===
        if final_score > self.best_score:
            print(f"  === NEW BEST CONFIGURATION ===")
            print(f"  Final Score: {final_score:.2f}")
            print(f"    Separation Score: {separation_score:.1f}/40")
            print(f"    Polarization Score: {polarization_score:.1f}/30") 
            print(f"    Consistency Score: {consistency_score:.1f}/20")
            print(f"    Bimodal Score: {bimodal_score:.1f}/10")
            print(f"  ")
            print(f"  Natural System Analysis:")
            print(f"    Overall range: {overall_min:.3f}V to {overall_max:.3f}V ({overall_range:.3f}V span)")
            print(f"    System center: {overall_mean:.3f}V ± {overall_std:.3f}V")
            print(f"  ")
            print(f"  Group Separation:")
            print(f"    ACTIVE: {active_mean:.3f}V ± {active_std:.3f}V  (range: {min(all_active_outputs):.3f}-{max(all_active_outputs):.3f}V)")
            print(f"    INACTIVE: {inactive_mean:.3f}V ± {inactive_std:.3f}V  (range: {min(all_inactive_outputs):.3f}-{max(all_inactive_outputs):.3f}V)")
            print(f"    Worst-case separation: {separation:.3f}V")
            print(f"    Mean separation: {mean_separation:.3f}V")
            if max_inactive > 0.001:
                er_linear = min_active / max_inactive
                print(f"    Extinction ratio: {10*np.log10(er_linear):.2f}dB (linear: {er_linear:.2f})")
            print(f"  ")
            print(f"  Voltage Distributions:")
            print(f"    ACTIVE outputs: {[f'{v:.3f}V' for v in all_active_outputs]}")
            print(f"    INACTIVE outputs: {[f'{v:.3f}V' for v in all_inactive_outputs[:6]]}... (showing first 6)")
            print(f"  ")
            print(f"  Test Pattern Results:")
            for i, result in enumerate(test_results):
                pattern = f"{int(result['input'][0] == V_MAX)}{int(result['input'][1] == V_MAX)}"
                outputs_str = [f'{o:.3f}' for o in result['outputs']]
                expected_ch = result['expected_active'] + 1
                print(f"    {pattern} -> Expected Ch{expected_ch}: {outputs_str}")
        
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
        
        kernel = (  
            ConstantKernel(1.0, constant_value_bounds=(0.1, 100)) *
            RBF(length_scale=1.0, length_scale_bounds=(0.1, 50.0)) +
            WhiteKernel(noise_level=0.1, noise_level_bounds=(0.00001, 10.0))
        )
        
        try:
            # Clear any existing GP to avoid conflicts
            self.gp = None
            
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
            print(f"    Continuing with random sampling...")
            self.gp = None
            return False

    def initial_sampling(self, n_samples=50):
        """Improved initial sampling with more diverse patterns"""
        print(f"Initial sampling with {n_samples} configurations...")
        
        configs = []
        
        # Diverse starting patterns based on successful logic gate approach
        good_patterns = [
            {h: 0.1 for h in MODIFIABLE_HEATERS},  # All low
            {h: 1.0 for h in MODIFIABLE_HEATERS},  # Low-medium
            {h: 2.5 for h in MODIFIABLE_HEATERS},  # Medium
            {h: 4.0 for h in MODIFIABLE_HEATERS},  # High
            {h: 4.9 for h in MODIFIABLE_HEATERS},  # Max
            {h: 0.1 if i < len(MODIFIABLE_HEATERS)//2 else 4.9 for i, h in enumerate(MODIFIABLE_HEATERS)},  # Half-half
            {h: random.choice([0.1, 1.0, 2.0, 3.0, 4.0, 4.9]) for h in MODIFIABLE_HEATERS},  # Random levels
            {h: 0.1 + (i % 10) * 0.5 for i, h in enumerate(MODIFIABLE_HEATERS)},  # Stepped pattern
            # Add more extreme patterns
            {h: random.choice([0.1, 4.9]) for h in MODIFIABLE_HEATERS},  # Binary extremes
            {h: 0.1 + (i / len(MODIFIABLE_HEATERS)) * 4.8 for i, h in enumerate(MODIFIABLE_HEATERS)},  # Linear gradient
        ]
        
        # Add good patterns (more of them)
        for pattern in good_patterns[:min(10, n_samples//3)]:
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
                print(f"Config {i+1}/{n_samples}: Score = {score:.2f}")
            else:
                print(f"Config {i+1}/{n_samples}: Score = {score:.2f}")
        
        print(f"Initial sampling complete. Working configs: {working_configs}/{len(self.y_evaluated)}")
        return working_configs

    def bayesian_optimize(self, n_iterations=15, batch_size=3):
        
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
                                
                # Smart beta selection based on progress
                if self.best_score < 0:
                    beta = 7.0    # High exploration when stuck in bad region
                elif self.best_score < 25:
                    beta = 5.0    # Moderate exploration for early progress  
                elif self.best_score < 70:
                    beta = 6.0    # Focus on refinement
                elif self.best_score < 90:
                    beta = 4.0    # Fine-tuning
                else:
                    beta = 2    # Pure refinement
                
                # Vectorized acquisition function
                ucb_scores = mu + beta * sigma
                
                print(f"Selected top {batch_size} candidates (β={beta}):")
                
                # Select the top candidates to actually test
                best_indices = np.argsort(ucb_scores)[-batch_size:]
                configs_to_test = [self.array_to_config(candidates[i]) for i in best_indices]
                
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
        print(f"Final best score: {self.best_score:.2f}")

    def explore_around_best(self):
        """Enhanced local exploration"""
        if not self.best_config:
            print("No best configuration available for local exploration.")
            return
        
        print(f"\nLocal exploration...")
        base_config = self.best_config.copy()
        
        # Multiple exploration strategies
        strategies = [
            {'name': 'coarse', 'range': 0.4, 'fraction': 0.4, 'samples': 10},    # Coarse search
            {'name': 'medium', 'range': 0.2, 'fraction': 0.3, 'samples': 15},   # Medium search
            {'name': 'fine', 'range': 0.1, 'fraction': 0.2, 'samples': 15},      # Fine tuning
        ]
        
        for strategy in strategies:
            print(f"\n  {strategy['name'].title()} local exploration (±{strategy['range']}V):")
            
            for i in range(strategy['samples']):
                new_config = base_config.copy()
                
                # Perturb specified fraction of heaters
                n_perturb = max(1, int(len(MODIFIABLE_HEATERS) * strategy['fraction']))
                heaters_to_perturb = random.sample(MODIFIABLE_HEATERS, n_perturb)
                
                for h in heaters_to_perturb:
                    current = new_config.get(h, 0.1)
                    perturbation = random.uniform(-strategy['range'], strategy['range'])
                    new_config[h] = max(V_MIN, min(V_MAX, current + perturbation))
                
                score = self.evaluate_configuration(new_config)
                self.add_evaluation(new_config, score)
                
                print(f"    {strategy['name']} {i+1}/{strategy['samples']}: Score = {score:.2f}")

    def test_final_configuration(self):
       
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
                    print(f"Excellent decoder performance!")
                elif logic_er_db >= 3.0:
                    print(f"Good decoder performance!")
                elif logic_er_db >= 1.0:
                    print(f"Marginal decoder performance")
                else:
                    print(f"Poor decoder performance")
            else:
                print(f"OVERLAPPING LOGIC LEVELS!")
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
        self.serial.close()
        self.scope.close()
        print("Connections closed.")

    def smart_reset(self):
        """Keep only the best data points and reset the rest"""
        if len(self.y_evaluated) < 20:
            return False
            
        # Keep top 20% of evaluations
        n_keep = max(5, len(self.y_evaluated) // 5)
        
        # Get indices of best scores
        best_indices = np.argsort(self.y_evaluated)[-n_keep:]
        
        # Reset to only best data
        self.X_evaluated = [self.X_evaluated[i] for i in best_indices]
        self.y_evaluated = [self.y_evaluated[i] for i in best_indices]
        
        print(f"Smart reset: kept {n_keep} best evaluations")
        self.fit_gaussian_process()
        return True

    def optimize(self):
        """Main optimization function"""
        #self.smart_reset()
        
        # Only do initial sampling if we don't have enough data
        if len(self.X_evaluated) < 30:
            needed_samples = 30 - len(self.X_evaluated)
            print(f"Need {needed_samples} more initial samples...")
            self.initial_sampling(n_samples=needed_samples)
        
        self.bayesian_optimize(n_iterations=15)
        self.explore_around_best()

        #self.bayesian_optimize(n_iterations=15)
        #self.explore_around_best()

        self.test_final_configuration()
        
        # Final save
        self.save_model()
        
        print(f"\nOptimization complete!")
        print(f"Best score: {self.best_score:.2f}")
        print(f"Total evaluations: {len(self.y_evaluated)}")
        print("\nFinal heater configuration:")
        print(self.format_config())
        
        return self.best_config, self.best_score


def main():
    start_time = time.time()
    
    optimizer = BayesianDecoderOptimizer()

    try:
        optimizer.optimize()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        print("Saving current state...")
        optimizer.save_model()
    except Exception as e:
        print(f"Error: {e}")
        print("Saving current state...")
        optimizer.save_model()
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