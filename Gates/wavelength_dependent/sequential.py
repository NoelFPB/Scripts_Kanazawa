import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc
import json
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
GATE_1548 = "AND"
GATE_1552 = "OR"
INPUT_HEATERS = [36, 37]
V_MIN = 0.1
V_MAX = 4.9
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in INPUT_HEATERS]
INPUT_COMBINATIONS = [(V_MIN, V_MIN), (V_MIN, V_MAX), (V_MAX, V_MIN), (V_MAX, V_MAX)]
LASER_ADDRESS = "GPIB0::6::INSTR"

def generate_truth_table(gate_type):
    """Generate truth table for given gate type"""
    tables = {
        "AND": [False, False, False, True],
        "OR": [False, True, True, True],
        "NAND": [True, True, True, False],
        "NOR": [True, False, False, False],
        "XOR": [False, True, True, False],
        "XNOR": [True, False, False, True]
    }
    
    return dict(zip(INPUT_COMBINATIONS, tables[gate_type]))

class HybridSequentialOptimizer:
    def __init__(self, gate_1548=GATE_1548, gate_1552=GATE_1552):
        self.gate_1548 = gate_1548
        self.gate_1552 = gate_1552
        self.truth_table_1548 = generate_truth_table(gate_1548)
        self.truth_table_1552 = generate_truth_table(gate_1552)
        
        print(f"Hybrid Sequential Optimization:")
        print(f"Phase 1: Master {gate_1548} gate at 1548nm")
        print(f"Phase 2: Master {gate_1552} gate at 1552nm") 
        print(f"Phase 3: Intelligent dual-wavelength fusion")
        
        # Phase 1: AND gate data
        self.X_and_phase = []
        self.y_and_phase = []
        self.gp_and = None
        self.best_and_configs = []
        
        # Phase 2: OR gate data  
        self.X_or_phase = []
        self.y_or_phase = []
        self.gp_or = None
        self.best_or_configs = []
        
        # Phase 3: Dual wavelength data
        self.X_dual_phase = []
        self.y_dual_and = []
        self.y_dual_or = []
        self.y_dual_combined = []
        
        # Final results
        self.best_config = None
        self.best_and_score = float('-inf')
        self.best_or_score = float('-inf')
        self.best_combined_score = float('-inf')
        self.best_details = {}
        
        # Base configuration
        self.base_config = {h: 0.01 for h in FIXED_FIRST_LAYER if h not in INPUT_HEATERS}
        
        # Initialize hardware
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        self.scope = self._init_scope()
        self.laser = self._init_laser()
        time.sleep(0.5)

    def _init_scope(self):
        rm = pyvisa.ResourceManager()
        scope = rm.open_resource(rm.list_resources()[0])
        scope.timeout = 5000
        scope.write(':CHANnel2:DISPlay ON')
        scope.write(':CHANnel2:SCALe 2')
        scope.write(':CHANnel2:OFFSet -6')
        return scope
    
    def _init_laser(self):
        rm = pyvisa.ResourceManager()
        laser = rm.open_resource(LASER_ADDRESS)
        laser.timeout = 5000
        laser.write_termination = ''
        laser.read_termination = ''
        return laser

    def config_to_array(self, config):
        return np.array([config[h] for h in MODIFIABLE_HEATERS])

    def array_to_config(self, x):
        return {h: x[i] for i, h in enumerate(MODIFIABLE_HEATERS)}

    def set_wavelength(self, wavelength_nm):
        self.laser.write(f'LW{wavelength_nm}nm')
        time.sleep(2)

    def turn_laser_on(self):
        self.laser.write('LE1')
        time.sleep(1)

    def turn_laser_off(self):
        self.laser.write('LE0')
        time.sleep(1)

    def send_heater_values(self, config):
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.serial.write(voltage_message.encode())
        self.serial.flush()
        time.sleep(0.01)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def measure_output(self):
        value = float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel2'))
        return round(value, 5)

    def calculate_extinction_ratio_score(self, high_outputs, low_outputs):
        if not high_outputs or not low_outputs:
            return -500
        
        min_high = min(high_outputs)
        max_low = max(low_outputs)
        logic_separation = min_high - max_low
        
        if logic_separation > 0:
            er_linear = min_high / max(max_low, 0.001)
            er_db = 10 * np.log10(er_linear)
            
            if er_db < 1:
                er_score = 0
            elif er_db < 3:
                er_score = 20 * (er_db - 1) / 2
            elif er_db < 5:
                er_score = 20 + 30 * (er_db - 3) / 2
            elif er_db < 7:
                er_score = 50 + 15 * (er_db - 5) / 2
            else:
                er_score = 65 + 5 * (1 - np.exp(-(er_db - 7) / 3))
        else:
            er_score = -30 * min(1.0, abs(logic_separation) / 0.5)
        
        # Signal strength score
        mean_high = sum(high_outputs) / len(high_outputs)
        strength_score = 20 * min(1.0, mean_high / 3.0)
        
        # Consistency score
        high_std = np.std(high_outputs) if len(high_outputs) > 1 else 0
        low_std = np.std(low_outputs) if len(low_outputs) > 1 else 0
        avg_std = (high_std + low_std) / 2
        consistency_score = 10 * np.exp(-avg_std * 5)
        
        total_score = er_score + strength_score + consistency_score
        
        # Penalties for output variations
        if len(high_outputs) > 1:
            high_range = max(high_outputs) - min(high_outputs)
            penalty = -30 * (high_range / 0.5) if high_range > 1 else -20 * (high_range / 0.5)
            total_score += penalty
        
        if len(low_outputs) > 1:
            low_range = max(low_outputs) - min(low_outputs)
            penalty = -30 * (low_range / 0.5) if low_range > 1 else -20 * (low_range / 0.5)
            total_score += penalty
        
        return min(100, max(-50, total_score))
    
    def evaluate_single_config_at_wavelength(self, config, truth_table):
        high_outputs = []
        low_outputs = []
        detailed_results = {}
        
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            expected_high = truth_table[input_state]
            self.send_heater_values(current_config)
            time.sleep(0.20)
            
            output = self.measure_output()
            detailed_results[input_state] = {'output': output, 'expected_high': expected_high}
            
            if expected_high:
                high_outputs.append(output)
            else:
                low_outputs.append(output)
        
        score = self.calculate_extinction_ratio_score(high_outputs, low_outputs)
        return score, detailed_results

    def fit_single_gate_gp(self, X_data, y_data, gate_name):
        """Fit GP model for single gate optimization"""
        if len(X_data) < 5:
            return None
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        print(f"    Fitting {gate_name} GP with {len(X)} points, score range: [{y.min():.1f}, {y.max():.1f}]")
        
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.1, 100)) *
            RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
            WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-5, 10.0))
        )
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        gp.fit(X, y)
        
        print(f"    {gate_name} GP fitted successfully!")
        return gp

    def suggest_next_single_gate_configs(self, gp_model, X_evaluated, y_evaluated, gate_name, batch_size=3):
        """
        Efficient candidate generation: Create 1500 candidates, test only top batch_size
        """
        if len(X_evaluated) < 5 or gp_model is None:
            print(f"  {gate_name} random sampling (insufficient data or no GP model)")
            configs = []
            for _ in range(batch_size):
                x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
                configs.append(self.array_to_config(x))
            return configs
        
        print(f"  {gate_name} generating and evaluating 1500 candidates...")
        
        # Generate 1500 diverse candidates
        candidates = []
        
        for _ in range(1500):
            if np.random.random() < 0.7:
                # 70% random exploration
                x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
            else:
                # 30% local search around best configs
                if y_evaluated:
                    best_indices = np.argsort(y_evaluated)[-min(5, len(y_evaluated)):]
                    base_idx = random.choice(best_indices)
                    base_x = X_evaluated[base_idx]
                    noise = np.random.normal(0, 0.5, len(MODIFIABLE_HEATERS))
                    x = np.clip(base_x + noise, V_MIN, V_MAX)
                else:
                    x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
            
            candidates.append(x)
        
        candidates = np.array(candidates)
        
        # Predict scores for ALL candidates (fast!)
        mu, sigma = gp_model.predict(candidates, return_std=True)
        
        # Adaptive beta
        best_score = max(y_evaluated) if y_evaluated else -1000
        if best_score < 50:
            beta = 3.0
        elif best_score < 60:
            beta = 2.0
        else:
            beta = 1.0
        
        print(f"    Using beta = {beta} based on best score {best_score:.1f}")
        
        ucb_scores = mu + beta * sigma
        
        # Select the top batch_size candidates
        best_indices = np.argsort(ucb_scores)[-batch_size:]
        configs = []
        
        print(f"    Selected top {batch_size} candidates:")
        for i, idx in enumerate(best_indices):
            configs.append(self.array_to_config(candidates[idx]))
            print(f"      Candidate {i+1}: Predicted={mu[idx]:.1f}±{sigma[idx]:.1f}, UCB={ucb_scores[idx]:.1f}")
        
        return configs

    def phase1_and_gate_optimization(self, n_iterations=20, batch_size=3):
        """Phase 1: Focus entirely on AND gate at 1548nm with efficient Bayesian optimization"""
        print(f"\n" + "="*60)
        print(f"PHASE 1: AND GATE MASTERY (1548nm)")
        print(f"="*60)
        
        # Set laser once and keep it there
        print("Setting laser to 1548nm for AND gate optimization...")
        self.set_wavelength(1548)
        self.turn_laser_on()
        time.sleep(14)
        
        # Initial diverse sampling
        print("Initial diverse sampling for AND gate...")
        
        configs = []
        
        # Add diverse starting patterns
        good_patterns = [
            {h: 0.1 for h in MODIFIABLE_HEATERS},
            {h: 1.0 for h in MODIFIABLE_HEATERS},
            {h: 2.5 for h in MODIFIABLE_HEATERS},
            {h: 4.0 for h in MODIFIABLE_HEATERS},
            {h: 4.9 for h in MODIFIABLE_HEATERS},
            {h: 0.1 if i < len(MODIFIABLE_HEATERS)//2 else 4.9 for i, h in enumerate(MODIFIABLE_HEATERS)},
            {h: random.choice([0.1, 1.0, 2.0, 3.0, 4.0, 4.9]) for h in MODIFIABLE_HEATERS},
            {h: 0.1 + (i % 10) * 0.5 for i, h in enumerate(MODIFIABLE_HEATERS)}
        ]
        
        for pattern in good_patterns[:min(6, 30//3)]:
            configs.append(pattern)
        
        # Latin Hypercube for remaining
        n_remaining = 36 - len(configs)
        if n_remaining > 0:
            sampler = qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS), seed=42)
            lhs_samples = sampler.random(n=n_remaining)
            
            for sample in lhs_samples:
                config = {}
                for j, h in enumerate(MODIFIABLE_HEATERS):
                    config[h] = V_MIN + sample[j] * (V_MAX - V_MIN)
                configs.append(config)
        
        # Evaluate initial configs
        for i, config in enumerate(configs):
            score, details = self.evaluate_single_config_at_wavelength(config, self.truth_table_1548)
            
            x = self.config_to_array(config)
            self.X_and_phase.append(x)
            self.y_and_phase.append(score)
            
            print(f"  Initial {i+1}: AND score {score:.1f}")
            
            if score > 20:
                self.best_and_configs.append((config.copy(), score))
        
        # Efficient Bayesian optimization
        print(f"\nBayesian optimization for AND gate ({n_iterations} iterations, {batch_size} configs/iteration)...")
        
        for iteration in range(n_iterations):
            print(f"\n--- AND Optimization Iteration {iteration + 1}/{n_iterations} ---")
            
            # Fit GP model
            if len(self.X_and_phase) >= 35:
                if self.gp_and is None or iteration % 5 == 0:
                    self.gp_and = self.fit_single_gate_gp(self.X_and_phase, self.y_and_phase, "AND")
            
            # Get batch of configs to test
            configs_to_test = self.suggest_next_single_gate_configs(
                self.gp_and, self.X_and_phase, self.y_and_phase, "AND", batch_size)
            
            # Test batch
            for i, config in enumerate(configs_to_test):
                score, details = self.evaluate_single_config_at_wavelength(config, self.truth_table_1548)
                
                x = self.config_to_array(config)
                self.X_and_phase.append(x)
                self.y_and_phase.append(score)
                
                print(f"    Config {i+1}/{batch_size}: AND score {score:.1f}")
                
                if score > 20:
                    self.best_and_configs.append((config.copy(), score))
                    if score > self.best_and_score:
                        self.best_and_score = score
                        print(f"      → NEW BEST AND SCORE!")
        
        # Sort best AND configs
        self.best_and_configs.sort(key=lambda x: x[1], reverse=True)
        
        best_and = max(self.y_and_phase) if self.y_and_phase else -1000
        print(f"\nPhase 1 Complete:")
        print(f"  Best AND score: {best_and:.1f}")
        print(f"  Good AND configs found: {len(self.best_and_configs)}")
        print(f"  Total evaluations: {len(self.X_and_phase)}")
        
        self.turn_laser_off()

    def phase2_or_gate_optimization(self, n_iterations=20, batch_size=3):
        """Phase 2: Focus entirely on OR gate at 1552nm with efficient Bayesian optimization"""
        print(f"\n" + "="*60)
        print(f"PHASE 2: OR GATE MASTERY (1552nm)")
        print(f"="*60)
        
        # Set laser once and keep it there
        print("Setting laser to 1552nm for OR gate optimization...")
        self.set_wavelength(1552)
        self.turn_laser_on()
        time.sleep(14)
        
        # Initial diverse sampling
        print("Initial diverse sampling for OR gate...")
        
        configs = []
        
        # Add diverse starting patterns
        good_patterns = [
            {h: 0.1 for h in MODIFIABLE_HEATERS},
            {h: 1.0 for h in MODIFIABLE_HEATERS},
            {h: 2.5 for h in MODIFIABLE_HEATERS},
            {h: 4.0 for h in MODIFIABLE_HEATERS},
            {h: 4.9 for h in MODIFIABLE_HEATERS},
            {h: 0.1 if i < len(MODIFIABLE_HEATERS)//2 else 4.9 for i, h in enumerate(MODIFIABLE_HEATERS)},
            {h: random.choice([0.1, 1.0, 2.0, 3.0, 4.0, 4.9]) for h in MODIFIABLE_HEATERS},
            {h: 0.1 + (i % 10) * 0.5 for i, h in enumerate(MODIFIABLE_HEATERS)}
        ]
        
        for pattern in good_patterns[:min(6, 30//3)]:
            configs.append(pattern)
        
        # Latin Hypercube for remaining
        n_remaining = 36 - len(configs)
        if n_remaining > 0:
            sampler = qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS), seed=42)
            lhs_samples = sampler.random(n=n_remaining)
            
            for sample in lhs_samples:
                config = {}
                for j, h in enumerate(MODIFIABLE_HEATERS):
                    config[h] = V_MIN + sample[j] * (V_MAX - V_MIN)
                configs.append(config)
        
        # Evaluate initial configs
        for i, config in enumerate(configs):
            score, details = self.evaluate_single_config_at_wavelength(config, self.truth_table_1552)
            
            x = self.config_to_array(config)
            self.X_or_phase.append(x)
            self.y_or_phase.append(score)
            
            print(f"  Initial {i+1}: OR score {score:.1f}")
            
            if score > 20:
                self.best_or_configs.append((config.copy(), score))
        
        # Efficient Bayesian optimization
        print(f"\nBayesian optimization for OR gate ({n_iterations} iterations, {batch_size} configs/iteration)...")
        
        for iteration in range(n_iterations):
            print(f"\n--- OR Optimization Iteration {iteration + 1}/{n_iterations} ---")
            
            # Fit GP model
            if len(self.X_or_phase) >= 35:
                if self.gp_or is None or iteration % 5 == 0:
                    self.gp_or = self.fit_single_gate_gp(self.X_or_phase, self.y_or_phase, "OR")
            
            # Get batch of configs to test
            configs_to_test = self.suggest_next_single_gate_configs(
                self.gp_or, self.X_or_phase, self.y_or_phase, "OR", batch_size)
            
            # Test batch
            for i, config in enumerate(configs_to_test):
                score, details = self.evaluate_single_config_at_wavelength(config, self.truth_table_1552)
                
                x = self.config_to_array(config)
                self.X_or_phase.append(x)
                self.y_or_phase.append(score)
                
                print(f"    Config {i+1}/{batch_size}: OR score {score:.1f}")
                
                if score > 20:
                    self.best_or_configs.append((config.copy(), score))
                    if score > self.best_or_score:
                        self.best_or_score = score
                        print(f"      → NEW BEST OR SCORE!")
        
        # Sort best OR configs
        self.best_or_configs.sort(key=lambda x: x[1], reverse=True)
        
        best_or = max(self.y_or_phase) if self.y_or_phase else -1000
        print(f"\nPhase 2 Complete:")
        print(f"  Best OR score: {best_or:.1f}")
        print(f"  Good OR configs found: {len(self.best_or_configs)}")
        print(f"  Total evaluations: {len(self.X_or_phase)}")
        
        self.turn_laser_off()

    def phase3_intelligent_fusion(self, n_evaluations=30):
        """Phase 3: Intelligent dual-wavelength candidate generation and testing"""
        print(f"\n" + "="*60)
        print(f"PHASE 3: INTELLIGENT DUAL-WAVELENGTH FUSION")
        print(f"Target: {n_evaluations} dual-wavelength evaluations")
        print(f"="*60)
        
        # Generate smart candidate configurations
        fusion_candidates = []
        
        # Strategy 1: Test best single-gate configs at both wavelengths
        print("Strategy 1: Testing best single-gate configs...")
        for config, score in self.best_and_configs[:5]:
            fusion_candidates.append(("Best AND", config))
        for config, score in self.best_or_configs[:5]:
            fusion_candidates.append(("Best OR", config))
        
        # Strategy 2: GP-guided fusion candidates using 1500 candidates
        if self.gp_and and self.gp_or:
            print("Strategy 2: GP-guided candidate generation (1500 candidates)...")
            
            # Generate 1500 candidates and predict performance on both models
            test_candidates = []
            for _ in range(1500):
                if np.random.random() < 0.7:
                    # 70% random exploration
                    x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
                else:
                    # 30% local search around best configs
                    if np.random.random() < 0.5 and self.best_and_configs:
                        # Around best AND config
                        base_config = random.choice(self.best_and_configs[:3])[0]
                        base_x = self.config_to_array(base_config)
                        noise = np.random.normal(0, 0.5, len(MODIFIABLE_HEATERS))
                        x = np.clip(base_x + noise, V_MIN, V_MAX)
                    elif self.best_or_configs:
                        # Around best OR config
                        base_config = random.choice(self.best_or_configs[:3])[0]
                        base_x = self.config_to_array(base_config)
                        noise = np.random.normal(0, 0.5, len(MODIFIABLE_HEATERS))
                        x = np.clip(base_x + noise, V_MIN, V_MAX)
                    else:
                        x = np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))
                
                test_candidates.append(x)
            
            test_candidates = np.array(test_candidates)
            
            # Predict on both GPs
            mu_and, sigma_and = self.gp_and.predict(test_candidates, return_std=True)
            mu_or, sigma_or = self.gp_or.predict(test_candidates, return_std=True)
            
            fusion_scores = []
            for i in range(len(test_candidates)):
                # Conservative fusion score - both must be good
                fusion_score = min(mu_and[i], mu_or[i]) + 0.5 * (sigma_and[i] + sigma_or[i])
                fusion_scores.append(fusion_score)
            
            # Take top fusion candidates
            best_fusion_indices = np.argsort(fusion_scores)[-10:]
            for idx in best_fusion_indices:
                config = self.array_to_config(test_candidates[idx])
                fusion_candidates.append(("GP Fusion", config))
                print(f"  Fusion candidate: AND pred={mu_and[idx]:.1f}, OR pred={mu_or[idx]:.1f}")
        
        # Strategy 3: Interpolation between good configs
        print("Strategy 3: Interpolation between good configs...")
        if self.best_and_configs and self.best_or_configs:
            for i in range(5):
                and_config = random.choice(self.best_and_configs[:3])[0]
                or_config = random.choice(self.best_or_configs[:3])[0]
                
                alpha = random.uniform(0.3, 0.7)
                interpolated = {}
                for h in MODIFIABLE_HEATERS:
                    interpolated[h] = alpha * and_config[h] + (1-alpha) * or_config[h]
                    interpolated[h] = max(0.1, min(4.9, interpolated[h]))
                
                fusion_candidates.append(("Interpolation", interpolated))
        
        # Limit candidates and evaluate in batches
        selected_candidates = fusion_candidates[:n_evaluations]
        print(f"Testing {len(selected_candidates)} fusion candidates...")
        
        # Evaluate candidates in batches to minimize laser switching
        batch_size = 10
        for batch_start in range(0, len(selected_candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(selected_candidates))
            batch = selected_candidates[batch_start:batch_end]
            
            print(f"\nEvaluating fusion batch {batch_start//batch_size + 1}...")
            
            # Test batch at 1548nm (AND)
            self.set_wavelength(1548)
            self.turn_laser_on()
            time.sleep(14)
            
            batch_and_scores = []
            for strategy, config in batch:
                score, details = self.evaluate_single_config_at_wavelength(config, self.truth_table_1548)
                batch_and_scores.append(score)
                print(f"    {strategy}: AND={score:.1f}")
            
            self.turn_laser_off()
            time.sleep(1)
            
            # Test batch at 1552nm (OR)
            self.set_wavelength(1552)
            self.turn_laser_on()
            time.sleep(14)
            
            batch_or_scores = []
            for strategy, config in batch:
                score, details = self.evaluate_single_config_at_wavelength(config, self.truth_table_1552)
                batch_or_scores.append(score)
                print(f"    {strategy}: OR={score:.1f}")
            
            self.turn_laser_off()
            
            # Process results
            for i, (strategy, config) in enumerate(batch):
                and_score = batch_and_scores[i]
                or_score = batch_or_scores[i]
                
                # Combined scoring
                if and_score > 30 and or_score > 30:
                    combined_score = (and_score + or_score) * 1.5
                elif and_score > 0 and or_score > 0:
                    combined_score = (and_score + or_score) * 0.8
                else:
                    combined_score = max(and_score, or_score) * 0.3
                
                # Track in dual phase data
                x = self.config_to_array(config)
                self.X_dual_phase.append(x)
                self.y_dual_and.append(and_score)
                self.y_dual_or.append(or_score)
                self.y_dual_combined.append(combined_score)
                
                # Update best if better
                if combined_score > self.best_combined_score:
                    self.best_combined_score = combined_score
                    self.best_and_score = and_score
                    self.best_or_score = or_score
                    self.best_config = config.copy()
                    self.best_details = {
                        'strategy': strategy,
                        'and_score': and_score,
                        'or_score': or_score,
                        'combined_score': combined_score
                    }
                    print(f"      → NEW BEST DUAL CONFIG! AND={and_score:.1f}, OR={or_score:.1f}, Combined={combined_score:.1f}")

        print(f"\nPhase 3 Complete:")
        print(f"  Best dual-wavelength scores: AND={self.best_and_score:.1f}, OR={self.best_or_score:.1f}")
        print(f"  Combined score: {self.best_combined_score:.1f}")

    def test_final_configuration(self):
        if not self.best_config:
            print("No final configuration to test.")
            return
        
        print(f"\n" + "="*60)
        print(f"FINAL CONFIGURATION TEST")
        print(f"="*60)
        
        for wavelength, gate_name, truth_table in [
            (1548, self.gate_1548, self.truth_table_1548),
            (1552, self.gate_1552, self.truth_table_1552)
        ]:
            print(f"\nTesting at {wavelength}nm ({gate_name} gate):")
            self.set_wavelength(wavelength)
            self.turn_laser_on()
            time.sleep(14)
            
            high_outputs = []
            low_outputs = []
            
            for input_state in INPUT_COMBINATIONS:
                current_config = self.best_config.copy()
                current_config[INPUT_HEATERS[0]] = input_state[0]
                current_config[INPUT_HEATERS[1]] = input_state[1]
                
                self.send_heater_values(current_config)
                time.sleep(0.3)
                output = self.measure_output()
                expected = truth_table[input_state]
                
                if expected:
                    high_outputs.append(output)
                else:
                    low_outputs.append(output)
                
                print(f"  {input_state}: {output:.4f}V ({'HIGH' if expected else 'LOW'})")
            
            if high_outputs and low_outputs:
                min_high = min(high_outputs)
                max_low = max(low_outputs)
                separation = min_high - max_low
                
                if separation > 0:
                    er_db = 10 * np.log10(min_high / max(max_low, 0.001))
                    print(f"  ✅ {gate_name} WORKING: {er_db:.2f} dB extinction ratio")
                    print(f"     Separation: {separation:.3f}V")
                else:
                    print(f"  ❌ {gate_name} NOT WORKING: overlap {abs(separation):.3f}V")
            
            self.turn_laser_off()
        
        # Final assessment
        print(f"\n=== HYBRID SEQUENTIAL OPTIMIZATION RESULTS ===")
        both_working = self.best_and_score > 30 and self.best_or_score > 30
        if both_working:
            print(f"🎉 SUCCESS: Both gates working!")
            print(f"   AND: {self.best_and_score:.1f}")
            print(f"   OR: {self.best_or_score:.1f}")
            print(f"   Strategy: {self.best_details.get('strategy', 'Unknown')}")
        else:
            print(f"⚠️  Partial success:")
            print(f"   AND: {self.best_and_score:.1f}")
            print(f"   OR: {self.best_or_score:.1f}")

    def format_config(self):
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
        if not self.best_config:
            return
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'optimization_type': 'hybrid_sequential_efficient',
            'gate_1548': self.gate_1548,
            'gate_1552': self.gate_1552,
            'phase1_and_evaluations': len(self.X_and_phase),
            'phase2_or_evaluations': len(self.X_or_phase),
            'phase3_dual_evaluations': len(self.X_dual_phase),
            'best_and_score': float(self.best_and_score),
            'best_or_score': float(self.best_or_score),
            'best_combined_score': float(self.best_combined_score),
            'best_config': self.format_config(),
            'best_details': self.best_details,
            'phase_summaries': {
                'phase1_best_and': float(max(self.y_and_phase)) if self.y_and_phase else None,
                'phase2_best_or': float(max(self.y_or_phase)) if self.y_or_phase else None,
                'good_and_configs_found': len(self.best_and_configs),
                'good_or_configs_found': len(self.best_or_configs)
            }
        }
        
        filename = f"hybrid_sequential_efficient_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")

    def cleanup(self):
        self.laser.write('LE0')
        self.serial.close()
        print("Cleanup complete")



    def optimize(self):
        print("HYBRID SEQUENTIAL DUAL-WAVELENGTH OPTIMIZATION")
        print("Efficient Bayesian optimization with 1500 candidates per iteration")
        self.phase2_or_gate_optimization(n_iterations=50, batch_size=3)
        # Phase 1: Master AND gate (1548nm) with efficient Bayesian optimization
        self.phase1_and_gate_optimization(n_iterations=50, batch_size=3)
        
        # Phase 2: Master OR gate (1552nm) with efficient Bayesian optimization

        
        # # Phase 3: Intelligent fusion
        self.phase3_intelligent_fusion(n_evaluations=30)
        
        # Final testing
        self.test_final_configuration()
        
        print(f"\nFinal configuration: {self.format_config()}")
        self.save_results()
        self.cleanup()
        
        return self.best_config, self.best_and_score, self.best_or_score

def main():
    start_time = time.time()
    optimizer = HybridSequentialOptimizer(GATE_1548, GATE_1552)
    
    best_config, best_and, best_or = optimizer.optimize()
    
    if best_config:
        print(f"\nOPTIMIZATION COMPLETE!")
        print(f"Best AND score: {best_and:.2f}")
        print(f"Best OR score: {best_or:.2f}")
        both_working = best_and > 30 and best_or > 30
        print(f"Both gates working: {'YES' if both_working else 'NO'}")
        print(f"Total evaluations: Phase 1: {len(optimizer.X_and_phase)}, Phase 2: {len(optimizer.X_or_phase)}, Phase 3: {len(optimizer.X_dual_phase)}")
        print(f"Efficiency: Generated {15*1500 + 15*1500} candidates, tested {len(optimizer.X_and_phase) + len(optimizer.X_or_phase) + len(optimizer.X_dual_phase)}")
    else:
        print("OPTIMIZATION FAILED")
    
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main()