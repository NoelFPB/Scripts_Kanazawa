import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import json
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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

class ImprovedDualWavelengthOptimizer:
    def __init__(self, gate_1548=GATE_1548, gate_1552=GATE_1552):
        self.gate_1548 = gate_1548
        self.gate_1552 = gate_1552
        self.truth_table_1548 = generate_truth_table(gate_1548)
        self.truth_table_1552 = generate_truth_table(gate_1552)
        
        print(f"Optimizing: {gate_1548} at 1548nm, {gate_1552} at 1552nm")
        
        # Bayesian optimization data
        self.X_evaluated = []
        self.y_1548 = []
        self.y_1552 = []
        self.y_combined = []
        self.gp_1548 = None
        self.gp_1552 = None
        self.gp_combined = None
        self.scaler = StandardScaler()
        
        # Best configurations tracking
        self.best_config = None
        self.best_score = float('-inf')
        self.best_details = {}
        self.best_1548_only = None
        self.best_1552_only = None
        self.best_1548_score = float('-inf')
        self.best_1552_score = float('-inf')
        
        # History for analysis
        self.evaluation_history = []
        
        # Base configuration for fixed heaters
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

    def add_evaluation(self, config, score_1548, score_1552, score_combined, details=None):
        x = self.config_to_array(config)
        self.X_evaluated.append(x)
        self.y_1548.append(score_1548)
        self.y_1552.append(score_1552)
        self.y_combined.append(score_combined)
        
        # Store full evaluation details
        self.evaluation_history.append({
            'config': config.copy(),
            'scores': {
                '1548nm': score_1548,
                '1552nm': score_1552,
                'combined': score_combined
            },
            'details': details
        })

    def fit_gaussian_processes(self):
        if len(self.X_evaluated) < 5:
            return
        
        X = np.array(self.X_evaluated)
        
        # Normalize features - fit scaler if not already fitted
        try:
            X_scaled = self.scaler.transform(X)
        except:
            # Scaler not fitted yet, fit it now
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
        
        # Use Matern kernel which often works better for optimization
        kernel = (ConstantKernel(1.0, constant_value_bounds=(0.01, 100)) *
                 Matern(length_scale=1.0, length_scale_bounds=(0.01, 100.0), nu=2.5) +
                 WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0)))
        
        # Fit separate GPs for each wavelength
        self.gp_1548 = GaussianProcessRegressor(
            kernel=kernel.clone_with_theta(kernel.theta), 
            n_restarts_optimizer=15, 
            alpha=1e-6, 
            normalize_y=True,
            random_state=42
        )
        self.gp_1548.fit(X_scaled, np.array(self.y_1548))
        
        self.gp_1552 = GaussianProcessRegressor(
            kernel=kernel.clone_with_theta(kernel.theta), 
            n_restarts_optimizer=15, 
            alpha=1e-6, 
            normalize_y=True,
            random_state=43
        )
        self.gp_1552.fit(X_scaled, np.array(self.y_1552))
        
        # Combined GP
        self.gp_combined = GaussianProcessRegressor(
            kernel=kernel.clone_with_theta(kernel.theta), 
            n_restarts_optimizer=15, 
            alpha=1e-6, 
            normalize_y=True,
            random_state=44
        )
        self.gp_combined.fit(X_scaled, np.array(self.y_combined))

    def acquisition_function(self, x, mode='combined', beta=2.0):
        if self.gp_combined is None:
            return random.random()
        
        x = x.reshape(1, -1)
        
        # Check if scaler is fitted before using it
        try:
            x_scaled = self.scaler.transform(x)
        except:
            # If scaler not fitted, return random value
            return random.random()
        
        if mode == 'combined':
            mu, sigma = self.gp_combined.predict(x_scaled, return_std=True)
        elif mode == '1548':
            mu, sigma = self.gp_1548.predict(x_scaled, return_std=True)
        elif mode == '1552':
            mu, sigma = self.gp_1552.predict(x_scaled, return_std=True)
        else:
            # Multi-objective acquisition
            mu1, sigma1 = self.gp_1548.predict(x_scaled, return_std=True)
            mu2, sigma2 = self.gp_1552.predict(x_scaled, return_std=True)
            # Weighted combination favoring balance
            mu = 0.5 * mu1 + 0.5 * mu2
            sigma = np.sqrt(0.25 * sigma1**2 + 0.25 * sigma2**2)
        
        # Upper confidence bound with exploration bonus
        ucb = mu[0] + beta * sigma[0]
        
        # Add small random perturbation to break ties
        return ucb + random.random() * 0.01

    def suggest_next_configs(self, n_configs=5):
        if len(self.X_evaluated) < 10:
            # More exploration early on
            configs = []
            for _ in range(n_configs):
                x = np.random.uniform(0.1, 4.9, len(MODIFIABLE_HEATERS))
                configs.append(self.array_to_config(x))
            return configs
        
        configs = []
        modes = ['combined', '1548', '1552', 'multi'] * (n_configs // 4 + 1)
        
        for i in range(n_configs):
            mode = modes[i]
            
            # Generate diverse candidates
            candidates = []
            
            # Random exploration (reduced)
            candidates.extend(np.random.uniform(0.1, 4.9, size=(1000, len(MODIFIABLE_HEATERS))))
            
            # Exploitation around best points
            if self.best_config:
                for _ in range(500):
                    noise = np.random.normal(0, 0.5, len(MODIFIABLE_HEATERS))
                    candidate = np.clip(self.config_to_array(self.best_config) + noise, 0.1, 4.9)
                    candidates.append(candidate)
            
            if self.best_1548_only and mode in ['1548', 'multi']:
                for _ in range(300):
                    noise = np.random.normal(0, 0.3, len(MODIFIABLE_HEATERS))
                    candidate = np.clip(self.config_to_array(self.best_1548_only) + noise, 0.1, 4.9)
                    candidates.append(candidate)
            
            if self.best_1552_only and mode in ['1552', 'multi']:
                for _ in range(300):
                    noise = np.random.normal(0, 0.3, len(MODIFIABLE_HEATERS))
                    candidate = np.clip(self.config_to_array(self.best_1552_only) + noise, 0.1, 4.9)
                    candidates.append(candidate)
            
            # Sobol sequence for better coverage
            if i < n_configs // 2:
                sobol = qmc.Sobol(d=len(MODIFIABLE_HEATERS), scramble=True, seed=i)
                sobol_points = sobol.random(200)
                sobol_configs = V_MIN + sobol_points * (V_MAX - V_MIN)
                candidates.extend(sobol_configs)
            
            candidates = np.array(candidates)
            
            # Adaptive beta for exploration/exploitation balance
            progress = len(self.X_evaluated) / 100.0
            beta = 3.0 * (1 - progress) + 1.0 * progress
            
            acquisition_values = [self.acquisition_function(c, mode=mode, beta=beta) for c in candidates]
            best_idx = np.argmax(acquisition_values)
            
            config = self.array_to_config(candidates[best_idx])
            
            # Ensure diversity
            if configs and min(np.linalg.norm(self.config_to_array(config) - self.config_to_array(c)) 
                              for c in configs) < 0.5:
                # Too similar, pick a different one
                sorted_indices = np.argsort(acquisition_values)[::-1]
                for idx in sorted_indices[1:]:
                    alt_config = self.array_to_config(candidates[idx])
                    if not configs or min(np.linalg.norm(self.config_to_array(alt_config) - self.config_to_array(c)) 
                                         for c in configs) >= 0.5:
                        config = alt_config
                        break
            
            configs.append(config)
        
        return configs

    def calculate_extinction_ratio_score(self, high_outputs, low_outputs):
        if not high_outputs or not low_outputs:
            return -1000
        
        min_high = min(high_outputs)
        max_low = max(low_outputs)
        mean_high = np.mean(high_outputs)
        mean_low = np.mean(low_outputs)
        
        # Logic separation
        separation = min_high - max_low
        
        if separation > 0:
            # Extinction ratio in dB
            er_linear = min_high / max(max_low, 0.001)
            er_db = 10 * np.log10(er_linear)
            
            # Progressive scoring for ER
            if er_db < 1:
                er_score = er_db * 10  # 0-10 points for 0-1 dB
            elif er_db < 3:
                er_score = 10 + (er_db - 1) * 20  # 10-50 points for 1-3 dB
            elif er_db < 6:
                er_score = 50 + (er_db - 3) * 15  # 50-95 points for 3-6 dB
            elif er_db < 10:
                er_score = 95 + (er_db - 6) * 1.25  # 95-100 points for 6-10 dB
            else:
                er_score = 100 + (er_db - 10) * 0.5  # Bonus for >10 dB
            
            # Signal strength bonus (favor stronger signals)
            strength_bonus = 10 * np.tanh(mean_high / 2.5)
            
            # Consistency bonus
            high_std = np.std(high_outputs) if len(high_outputs) > 1 else 0
            low_std = np.std(low_outputs) if len(low_outputs) > 1 else 0
            consistency_penalty = -5 * (high_std + low_std)
            
            # Separation bonus
            separation_bonus = 5 * np.tanh(separation / 0.5)
            
            total_score = er_score + strength_bonus + separation_bonus + consistency_penalty
            
        else:
            # Overlapping levels - heavy penalty
            overlap = abs(separation)
            total_score = -50 - 100 * (overlap / 0.5)
            
            # Small bonus if means are well separated
            mean_separation = mean_high - mean_low
            if mean_separation > 0:
                total_score += 10 * np.tanh(mean_separation / 0.5)
        
        return max(-1000, min(150, total_score))

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

    def evaluate_single_config_at_wavelength(self, config, truth_table, wavelength):
        high_outputs = []
        low_outputs = []
        detailed_results = {}
        
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            expected_high = truth_table[input_state]
            self.send_heater_values(current_config)
            time.sleep(0.2)
            
            # Take multiple measurements for better accuracy
            measurements = []
            for _ in range(3):
                measurements.append(self.measure_output())
                time.sleep(0.05)
            
            output = np.median(measurements)
            detailed_results[input_state] = {
                'output': output, 
                'expected_high': expected_high,
                'measurements': measurements
            }
            
            if expected_high:
                high_outputs.append(output)
            else:
                low_outputs.append(output)
        
        score = self.calculate_extinction_ratio_score(high_outputs, low_outputs)
        return score, detailed_results

    def evaluate_configs_dual_wavelength(self, configs):
        print(f"\nDual-wavelength evaluation of {len(configs)} configurations...")
        
        results = []
        
        # Evaluate at 1548nm
        print(f"Setting laser to 1548nm for {self.gate_1548} gate testing...")
        self.set_wavelength(1548)
        self.turn_laser_on()
        time.sleep(14)
        
        results_1548 = []
        for i, config in enumerate(configs):
            score, details = self.evaluate_single_config_at_wavelength(
                config, self.truth_table_1548, 1548)
            results_1548.append((score, details))
            print(f"  Config {i+1}/1548nm: Score {score:.1f}")
            
            # Track best single-wavelength configs
            if score > self.best_1548_score:
                self.best_1548_score = score
                self.best_1548_only = config.copy()
        
        self.turn_laser_off()
        
        # Evaluate at 1552nm
        print(f"Setting laser to 1552nm for {self.gate_1552} gate testing...")
        self.set_wavelength(1552)
        self.turn_laser_on()
        time.sleep(14)
        
        results_1552 = []
        for i, config in enumerate(configs):
            score, details = self.evaluate_single_config_at_wavelength(
                config, self.truth_table_1552, 1552)
            results_1552.append((score, details))
            print(f"  Config {i+1}/1552nm: Score {score:.1f}")
            
            # Track best single-wavelength configs
            if score > self.best_1552_score:
                self.best_1552_score = score
                self.best_1552_only = config.copy()
        
        self.turn_laser_off()
        
        # Combine results
        for i, config in enumerate(configs):
            score_1548, details_1548 = results_1548[i]
            score_1552, details_1552 = results_1552[i]
            
            # Improved combined scoring
            if score_1548 < -500 or score_1552 < -500:
                # Complete failure
                combined_score = -2000
            else:
                # Normalize negative scores
                norm_1548 = max(0, score_1548 + 50) / 150
                norm_1552 = max(0, score_1552 + 50) / 150
                
                # Geometric mean favors balanced performance
                if norm_1548 > 0 and norm_1552 > 0:
                    geometric_mean = np.sqrt(norm_1548 * norm_1552)
                    combined_score = geometric_mean * 200
                    
                    # Bonus for good performance on both
                    if score_1548 > 50 and score_1552 > 50:
                        combined_score *= 1.2
                else:
                    combined_score = max(score_1548, score_1552) * 0.3
            
            # Track results
            self.add_evaluation(config, score_1548, score_1552, combined_score, {
                '1548nm': details_1548,
                '1552nm': details_1552
            })
            
            # Update best configuration
            if combined_score > self.best_score:
                self.best_score = combined_score
                self.best_config = config.copy()
                self.best_details = {
                    '1548nm': details_1548,
                    '1552nm': details_1552,
                    'scores': {
                        '1548nm': score_1548, 
                        '1552nm': score_1552, 
                        'combined': combined_score
                    }
                }
                print(f"  *** NEW BEST: 1548nm={score_1548:.1f}, 1552nm={score_1552:.1f}, Combined={combined_score:.1f} ***")
            
            results.append((config, combined_score, {
                '1548nm': (score_1548, details_1548),
                '1552nm': (score_1552, details_1552)
            }))
        
        return results

    def set_wavelength(self, wavelength_nm):
        self.laser.write(f'LW{wavelength_nm}nm')
        time.sleep(0.5)

    def turn_laser_on(self):
        self.laser.write('LE1')
        time.sleep(0.5)

    def turn_laser_off(self):
        self.laser.write('LE0')
        time.sleep(0.5)

    def initial_sampling(self, n_samples=30):
        print(f"\nInitial sampling with {n_samples} configurations")
        
        configs = []
        
        # Include some structured initial points
        configs.append({h: 0.1 for h in MODIFIABLE_HEATERS})  # All low
        configs.append({h: 4.9 for h in MODIFIABLE_HEATERS})  # All high
        configs.append({h: 2.5 for h in MODIFIABLE_HEATERS})  # All middle
        
        # Latin Hypercube Sampling for good coverage
        sampler = qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS), seed=42)
        samples = sampler.random(n=n_samples-3)
        
        for sample in samples:
            config = {}
            for j, h in enumerate(MODIFIABLE_HEATERS):
                config[h] = V_MIN + sample[j] * (V_MAX - V_MIN)
            configs.append(config)
        
        self.evaluate_configs_dual_wavelength(configs)
        print(f"Initial sampling complete. Best score: {self.best_score:.1f}")

    def differential_evolution_phase(self, n_generations=10):
        print(f"\nDifferential evolution phase ({n_generations} generations)")
        
        def objective(x):
            # Use GP predictions if available
            if self.gp_combined is not None and len(self.X_evaluated) > 20:
                try:
                    x_scaled = self.scaler.transform(x.reshape(1, -1))
                    pred, _ = self.gp_combined.predict(x_scaled, return_std=True)
                    return -pred[0]  # Minimize negative score
                except:
                    return random.random()  # Random if scaler not fitted
            else:
                return random.random()  # Random if no GP yet
        
        bounds = [(V_MIN, V_MAX)] * len(MODIFIABLE_HEATERS)
        
        for gen in range(n_generations):
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=20,
                popsize=10,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=gen
            )
            
            config = self.array_to_config(result.x)
            self.evaluate_configs_dual_wavelength([config])
            
            if len(self.X_evaluated) >= 10:
                self.fit_gaussian_processes()

    def bayesian_optimize(self, n_iterations=50, batch_size=5):
        print(f"\nBayesian optimization: {n_iterations} iterations, batch size {batch_size}")
        
        iteration = 0
        no_improvement_count = 0
        last_best_score = self.best_score
        
        while iteration < n_iterations:
            # Adaptive batch size
            if no_improvement_count > 5:
                current_batch_size = min(batch_size + 2, 10)
            else:
                current_batch_size = batch_size
            
            # Get suggestions
            batch_configs = self.suggest_next_configs(
                min(current_batch_size, n_iterations - iteration))
            iteration += len(batch_configs)
            
            # Evaluate
            results = self.evaluate_configs_dual_wavelength(batch_configs)
            
            # Update GP models
            if len(self.X_evaluated) >= 10:
                self.fit_gaussian_processes()
            
            # Check for improvement
            if self.best_score > last_best_score:
                no_improvement_count = 0
                last_best_score = self.best_score
            else:
                no_improvement_count += 1
            
            # Periodic focused search if stuck
            if no_improvement_count > 10 and self.best_config:
                print("  Performing focused local search...")
                self.focused_search_around_best(n_variations=5)
                no_improvement_count = 0

    def focused_search_around_best(self, n_variations=10):
        if not self.best_config:
            return
        
        print(f"\nFocused search with {n_variations} variations around best config")
        variations = []
        
        # Different perturbation strategies
        for i in range(n_variations):
            variation = self.best_config.copy()
            
            if i < n_variations // 3:
                # Small perturbations
                for h in MODIFIABLE_HEATERS:
                    if random.random() < 0.3:
                        variation[h] = np.clip(
                            variation[h] + random.uniform(-0.1, 0.1), 
                            V_MIN, V_MAX
                        )
            elif i < 2 * n_variations // 3:
                # Medium perturbations on subset
                n_modify = random.randint(3, 8)
                heaters_to_modify = random.sample(MODIFIABLE_HEATERS, n_modify)
                for h in heaters_to_modify:
                    variation[h] = np.clip(
                        variation[h] + random.uniform(-0.3, 0.3), 
                        V_MIN, V_MAX
                    )
            else:
                # Larger coordinated changes
                direction = np.random.normal(0, 0.2, len(MODIFIABLE_HEATERS))
                direction = direction / np.linalg.norm(direction)
                step_size = random.uniform(0.2, 0.5)
                
                for j, h in enumerate(MODIFIABLE_HEATERS):
                    variation[h] = np.clip(
                        variation[h] + step_size * direction[j], 
                        V_MIN, V_MAX
                    )
            
            variations.append(variation)
        
        self.evaluate_configs_dual_wavelength(variations)

    def test_final_configuration(self):
        if not self.best_config:
            print("No configuration to test.")
            return
        
        print("\n" + "="*60)
        print("FINAL CONFIGURATION TESTING")
        print("="*60)
        
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
                
                # Multiple measurements for final test
                measurements = []
                for _ in range(5):
                    measurements.append(self.measure_output())
                    time.sleep(0.05)
                
                output = np.median(measurements)
                std = np.std(measurements)
                expected = truth_table[input_state]
                
                if expected:
                    high_outputs.append(output)
                else:
                    low_outputs.append(output)
                
                print(f"  {input_state}: {output:.4f}V ±{std:.4f} ({'HIGH' if expected else 'LOW'})")
            
            if high_outputs and low_outputs:
                min_high = min(high_outputs)
                max_low = max(low_outputs)
                mean_high = np.mean(high_outputs)
                mean_low = np.mean(low_outputs)
                separation = min_high - max_low
                
                if separation > 0:
                    er_db = 10 * np.log10(min_high / max(max_low, 0.001))
                    print(f"\n  Performance Metrics:")
                    print(f"    Extinction Ratio: {er_db:.2f} dB")
                    print(f"    Logic Separation: {separation:.3f}V")
                    print(f"    Mean High: {mean_high:.3f}V")
                    print(f"    Mean Low: {mean_low:.3f}V")
                else:
                    print(f"\n  WARNING: OVERLAPPING LOGIC LEVELS!")
                    print(f"    Overlap: {abs(separation):.3f}V")
                    print(f"    Mean separation: {mean_high - mean_low:.3f}V")
            
            self.turn_laser_off()

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
        
        # Analyze performance across evaluations
        performance_summary = {
            'total_evaluations': len(self.evaluation_history),
            'best_scores': {
                '1548nm_only': float(self.best_1548_score),
                '1552nm_only': float(self.best_1552_score),
                'combined': float(self.best_score)
            },
            'convergence_history': [
                {
                    'iteration': i,
                    'scores': hist['scores']
                } for i, hist in enumerate(self.evaluation_history[-20:])  # Last 20
            ]
        }
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'gate_1548': self.gate_1548,
            'gate_1552': self.gate_1552,
            'best_score': float(self.best_score),
            'best_config': self.format_config(),
            'best_details': serialize_for_json(self.best_details) if self.best_details else {},
            'performance_summary': performance_summary,
            'best_single_wavelength_configs': {
                '1548nm': self.format_config() if self.best_1548_only else {},
                '1552nm': self.format_config() if self.best_1552_only else {}
            }
        }
        
        filename = f"improved_dual_wavelength_{self.gate_1548}_{self.gate_1552}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
            print("\nFinal configuration (copy this):")
            print(json.dumps(self.format_config(), indent=2))

    def cleanup(self):
        self.laser.write('LE0')
        self.serial.close()
        print("Cleanup complete")

    def optimize(self):
        print("\n" + "="*60)
        print("IMPROVED DUAL-WAVELENGTH OPTIMIZATION")
        print(f"Target: {self.gate_1548} @ 1548nm, {self.gate_1552} @ 1552nm")
        print("="*60)
        
        start_time = time.time()
        
        # Phase 1: Initial exploration
        self.initial_sampling(n_samples=40)
        
        # Phase 2: Early Bayesian optimization
        self.bayesian_optimize(n_iterations=30, batch_size=5)
        
        # # Phase 3: Differential evolution for global search
        # if len(self.X_evaluated) > 20: 
        #     self.differential_evolution_phase(n_generations=5)
        
        # # Phase 4: Intensive Bayesian optimization
        # self.bayesian_optimize(n_iterations=40, batch_size=8)
        
        # # Phase 5: Final focused refinement
        # self.focused_search_around_best(n_variations=15)
        
        # Phase 6: Last push with smaller perturbations
        if self.best_config:
            print("\nFinal refinement phase...")
            final_configs = []
            for _ in range(10):
                config = self.best_config.copy()
                # Very small perturbations
                for h in MODIFIABLE_HEATERS:
                    if random.random() < 0.2:
                        config[h] = np.clip(
                            config[h] + random.uniform(-0.05, 0.05),
                            V_MIN, V_MAX
                        )
                final_configs.append(config)
            self.evaluate_configs_dual_wavelength(final_configs)
        
        # Final testing
        self.test_final_configuration()
        
        elapsed_time = (time.time() - start_time) / 60
        
        print(f"\n" + "="*60)
        print(f"OPTIMIZATION COMPLETE!")
        print(f"Total evaluations: {len(self.evaluation_history)}")
        print(f"Best combined score: {self.best_score:.2f}")
        print(f"Best single wavelength scores:")
        print(f"  1548nm: {self.best_1548_score:.2f}")
        print(f"  1552nm: {self.best_1552_score:.2f}")
        print(f"Time elapsed: {elapsed_time:.1f} minutes")
        print(f"="*60)
        
        self.save_results()
        self.cleanup()
        
        return self.best_config, self.best_score

def main():
    optimizer = ImprovedDualWavelengthOptimizer(GATE_1548, GATE_1552)
    best_config, best_score = optimizer.optimize()
    
    if best_config:
        print(f"\nFINAL RESULT: Best combined score = {best_score:.2f}")
    else:
        print("\nOPTIMIZATION FAILED")

if __name__ == "__main__":
    main()       