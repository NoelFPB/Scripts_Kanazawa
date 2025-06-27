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

class DualWavelengthOptimizer:
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
        self.gp_combined = None
        
        # Best configuration
        self.best_config = None
        self.best_score = float('-inf')
        self.best_details = {}
        
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

    def add_evaluation(self, config, score_1548, score_1552, score_combined):
        x = self.config_to_array(config)
        self.X_evaluated.append(x)
        self.y_1548.append(score_1548)
        self.y_1552.append(score_1552)
        self.y_combined.append(score_combined)

    def fit_gaussian_processes(self):
        if len(self.X_evaluated) < 5:
            return
        
        X = np.array(self.X_evaluated)
        y = np.array(self.y_combined)
        
        kernel = (ConstantKernel(1.0, constant_value_bounds=(0.1, 100)) *
                 RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
                 WhiteKernel(noise_level=0.3, noise_level_bounds=(1e-5, 10.0)))
        
        self.gp_combined = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, random_state=42)
        self.gp_combined.fit(X, y)

    def acquisition_function(self, x):
        if self.gp_combined is None:
            return random.random()
        
        x = x.reshape(1, -1)
        mu, sigma = self.gp_combined.predict(x, return_std=True)
        beta = 5.0
        return mu[0] + beta * sigma[0] + random.random() * 0.1

    def suggest_next_config(self):
        if len(self.X_evaluated) < 5:
            x = np.random.uniform(0.1, 4.9, len(MODIFIABLE_HEATERS))
            return self.array_to_config(x)
        
        # Generate candidates
        candidates = []
        
        # Random exploration
        candidates.extend(np.random.uniform(0.1, 4.9, size=(2000, len(MODIFIABLE_HEATERS))))
        
        # Local search around best points
        best_configs = sorted(zip(self.X_evaluated, self.y_combined), 
                            key=lambda x: x[1], reverse=True)[:5]
        for _ in range(2000):
            base_x, _ = random.choice(best_configs)
            noise = np.random.normal(0, 0.3, len(MODIFIABLE_HEATERS))
            candidate = np.clip(base_x + noise, 0.1, 4.9)
            candidates.append(candidate)
        
        # Grid-based exploration
        for _ in range(1000):
            candidate = [random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]) 
                        for _ in range(len(MODIFIABLE_HEATERS))]
            candidates.append(np.array(candidate))
        
        candidates = np.array(candidates)
        acquisition_values = [self.acquisition_function(c) for c in candidates]
        best_idx = np.argmax(acquisition_values)
        
        return self.array_to_config(candidates[best_idx])

    def set_wavelength(self, wavelength_nm):
        self.laser.write(f'LW{wavelength_nm}nm')
        time.sleep(0.5)

    def turn_laser_on(self):
        self.laser.write('LE1')
        time.sleep(0.5)

    def turn_laser_off(self):
        self.laser.write('LE0')
        time.sleep(0.5)

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
            
            output = self.measure_output()
            detailed_results[input_state] = {'output': output, 'expected_high': expected_high}
            
            if expected_high:
                high_outputs.append(output)
            else:
                low_outputs.append(output)
        
        score = self.calculate_extinction_ratio_score(high_outputs, low_outputs)
        return score, detailed_results

    def evaluate_single_wavelength_batch(self, configs, wavelength, truth_table, gate_name):
        print(f"Setting laser to {wavelength}nm for {gate_name} gate testing...")
        self.set_wavelength(wavelength)
        self.turn_laser_on()
        time.sleep(14)
        
        results = []
        for i, config in enumerate(configs):
            score, details = self.evaluate_single_config_at_wavelength(config, truth_table, wavelength)
            results.append((config, score, details))
            print(f"  Config {i+1}: Score {score:.1f}")
        
        self.turn_laser_off()
        return results

    def evaluate_configs_dual_wavelength(self, configs):
        print(f"Dual-wavelength evaluation of {len(configs)} configurations...")
        
        # Evaluate at both wavelengths
        results_1548 = self.evaluate_single_wavelength_batch(
            configs, 1548, self.truth_table_1548, self.gate_1548)
        results_1552 = self.evaluate_single_wavelength_batch(
            configs, 1552, self.truth_table_1552, self.gate_1552)
        
        combined_results = []
        for i, config in enumerate(configs):
            _, score_1548, details_1548 = results_1548[i]
            _, score_1552, details_1552 = results_1552[i]
            
            # Combined scoring
            if score_1548 < -500 or score_1552 < -500:
                combined_score = -2000
            else:
                positive_1548 = max(0, score_1548)
                positive_1552 = max(0, score_1552)
                
                if positive_1548 > 0 and positive_1552 > 0:
                    combined_score = (positive_1548 + positive_1552) * 1.5
                elif positive_1548 > 0 or positive_1552 > 0:
                    combined_score = max(positive_1548, positive_1552) * 0.6
                else:
                    combined_score = max(score_1548, score_1552) * 0.3
            
            # Update best configuration
            if combined_score > self.best_score:
                self.best_score = combined_score
                self.best_config = config.copy()
                self.best_details = {
                    '1548nm': details_1548,
                    '1552nm': details_1552,
                    'scores': {'1548nm': score_1548, '1552nm': score_1552, 'combined': combined_score}
                }
                print(f"  NEW BEST: 1548nm={score_1548:.1f}, 1552nm={score_1552:.1f}, Combined={combined_score:.1f}")
            
            combined_results.append((config, combined_score, {
                '1548nm': (score_1548, details_1548),
                '1552nm': (score_1552, details_1552)
            }))
        
        return combined_results

    def initial_sampling(self, n_samples=20):
        print(f"Initial sampling with {n_samples} configurations")
        
        configs = []
        configs.append({h: 0.1 for h in MODIFIABLE_HEATERS})  # Zero config
        
        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS), seed=42)
        samples = sampler.random(n=n_samples-1)
        
        for sample in samples:
            config = {}
            for j, h in enumerate(MODIFIABLE_HEATERS):
                config[h] = V_MIN + sample[j] * (V_MAX - V_MIN)
            configs.append(config)
        
        self.evaluate_configs_dual_wavelength(configs)
        print(f"Initial sampling complete. Best score: {self.best_score:.1f}")

    def bayesian_optimize(self, n_iterations=25, batch_size=5):
        print(f"Bayesian optimization: {n_iterations} iterations, batch size {batch_size}")
        
        iteration = 0
        while iteration < n_iterations:
            batch_configs = []
            
            for _ in range(min(batch_size, n_iterations - iteration)):
                config = self.suggest_next_config()
                batch_configs.append(config)
                iteration += 1
            
            results = self.evaluate_configs_dual_wavelength(batch_configs)
            
            # Add results to Bayesian optimizer
            for config, combined_score, details in results:
                score_1548 = details['1548nm'][0] if '1548nm' in details else -1000
                score_1552 = details['1552nm'][0] if '1552nm' in details else -1000
                self.add_evaluation(config, score_1548, score_1552, combined_score)
            
            if len(self.X_evaluated) >= 3:
                self.fit_gaussian_processes()

    def focused_search_around_best(self, n_variations=12):
        if not self.best_config:
            return
        
        print(f"Focused search with {n_variations} variations")
        variations = []
        
        for _ in range(n_variations):
            variation = self.best_config.copy()
            heaters_to_modify = random.sample(MODIFIABLE_HEATERS, 
                                            random.randint(1, min(5, len(MODIFIABLE_HEATERS))))
            
            for h in heaters_to_modify:
                current = variation[h]
                perturbation = random.uniform(-0.2, 0.2)
                variation[h] = max(0.1, min(4.9, current + perturbation))
            
            variations.append(variation)
        
        self.evaluate_configs_dual_wavelength(variations)

    def test_final_configuration(self):
        if not self.best_config:
            print("No configuration to test.")
            return
        
        print("\nTesting final dual-wavelength configuration")
        
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
                    print(f"  Extinction Ratio: {er_db:.2f} dB")
                    print(f"  Separation: {separation:.3f}V")
                else:
                    print(f"  OVERLAPPING LOGIC LEVELS (overlap: {abs(separation):.3f}V)")
            
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
        self.laser.write('LE0')
        self.serial.close()
        print("Cleanup complete")

    def optimize(self):
        print("DUAL-WAVELENGTH OPTIMIZATION")
        
        self.initial_sampling(n_samples=60)
        self.bayesian_optimize(n_iterations=50, batch_size=10)
        #self.focused_search_around_best(n_variations=10)
        #self.bayesian_optimize(n_iterations=30, batch_size=10)
        #self.focused_search_around_best(n_variations=10)
        self.test_final_configuration()
        
        print(f"Final configuration: {self.format_config()}")
        self.save_results()
        self.cleanup()
        
        return self.best_config, self.best_score

def main():
    start_time = time.time()
    optimizer = DualWavelengthOptimizer(GATE_1548, GATE_1552)
    
    best_config, best_score = optimizer.optimize()
    
    if best_config:
        print(f"OPTIMIZATION COMPLETE! Best score: {best_score:.2f}")
    else:
        print("OPTIMIZATION FAILED")
    
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main()