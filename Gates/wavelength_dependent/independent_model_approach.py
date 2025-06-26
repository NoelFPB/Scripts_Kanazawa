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

class DualModelOptimizer:
    def __init__(self, gate_1548=GATE_1548, gate_1552=GATE_1552):
        self.gate_1548 = gate_1548
        self.gate_1552 = gate_1552
        self.truth_table_1548 = generate_truth_table(gate_1548)
        self.truth_table_1552 = generate_truth_table(gate_1552)
        
        print(f"Dual Model Optimization: {gate_1548} at 1548nm, {gate_1552} at 1552nm")
        
        # Separate Bayesian optimization data for each gate
        self.X_evaluated = []           # Shared input configurations
        self.y_1548_and = []           # AND gate scores at 1548nm
        self.y_1552_or = []            # OR gate scores at 1552nm
        
        # Separate Gaussian Process models
        self.gp_and = None             # Model for AND gate performance
        self.gp_or = None              # Model for OR gate performance
        
        # Best configuration tracking
        self.best_config = None
        self.best_and_score = float('-inf')
        self.best_or_score = float('-inf')
        self.best_combined_score = float('-inf')
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

    def add_evaluation(self, config, score_and, score_or):
        """Add evaluation to both separate models"""
        x = self.config_to_array(config)
        self.X_evaluated.append(x)
        self.y_1548_and.append(score_and)
        self.y_1552_or.append(score_or)

    def fit_gaussian_processes(self):
        """Fit separate GP models for AND and OR gates"""
        if len(self.X_evaluated) < 5:
            return
        
        X = np.array(self.X_evaluated)
        y_and = np.array(self.y_1548_and)
        y_or = np.array(self.y_1552_or)
        
        print(f"    Fitting separate GPs with {len(X)} points")
        print(f"    AND scores range: [{y_and.min():.1f}, {y_and.max():.1f}]")
        print(f"    OR scores range: [{y_or.min():.1f}, {y_or.max():.1f}]")
        
        # Same kernel for both models
        kernel = (ConstantKernel(1.0, constant_value_bounds=(0.1, 100)) *
                 RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
                 WhiteKernel(noise_level=0.3, noise_level_bounds=(0.01, 10.0)))
        
        # Fit AND gate model
        self.gp_and = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, random_state=42)
        self.gp_and.fit(X, y_and)
        
        # Fit OR gate model  
        self.gp_or = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, random_state=43)
        self.gp_or.fit(X, y_or)
        
        print(f"    Both GP models fitted successfully!")

    def balanced_acquisition_function(self, x):
        """
        Balanced acquisition function prioritizing both gates working
        Strategy: Use min-max with some exploration bonus
        """
        if self.gp_and is None or self.gp_or is None:
            return random.random()
        
        x = x.reshape(1, -1)
        
        # Get predictions from both models
        mu_and, sigma_and = self.gp_and.predict(x, return_std=True)
        mu_or, sigma_or = self.gp_or.predict(x, return_std=True)
        
        mu_and, sigma_and = mu_and[0], sigma_and[0]
        mu_or, sigma_or = mu_or[0], sigma_or[0]
        
        # Upper Confidence Bound for each gate
        beta = 3.0  # Moderate exploration
        ucb_and = mu_and + beta * sigma_and
        ucb_or = mu_or + beta * sigma_or
        
        # BALANCED ACQUISITION STRATEGIES:
        
        # 1. Min-max approach (conservative - both must be good)
        min_max_score = min(ucb_and, ucb_or)
        
        # 2. Geometric mean (balanced - penalizes if one is terrible)
        # Shift to positive range first - use dynamic shift based on actual values
        min_score = min(ucb_and, ucb_or)
        shift_amount = max(100, abs(min_score) + 10)  # Ensure positive values
        shifted_and = ucb_and + shift_amount
        shifted_or = ucb_or + shift_amount
        geom_mean_score = np.sqrt(shifted_and * shifted_or) - shift_amount
        
        # 3. Harmonic mean (heavily penalizes poor performance in either)
        if ucb_and > 0 and ucb_or > 0:
            harm_mean_score = 2 * ucb_and * ucb_or / (ucb_and + ucb_or)
        else:
            harm_mean_score = min(ucb_and, ucb_or)
        
        # 4. Working threshold bonus
        working_threshold = 30  # Score needed to "work"
        both_working_bonus = 0
        if mu_and > working_threshold and mu_or > working_threshold:
            both_working_bonus = 50  # Big bonus for both working
        elif mu_and > working_threshold or mu_or > working_threshold:
            both_working_bonus = 20  # Small bonus for one working
        
        # Combine strategies (weighted combination)
        final_score = (
            0.4 * min_max_score +           # Conservative base
            0.3 * geom_mean_score +         # Balanced performance
            0.2 * harm_mean_score +         # Penalty for poor performance
            0.1 * both_working_bonus        # Bonus for working gates
        )
        
        # Add small random tie-breaker
        final_score += random.random() * 0.1
        
        return final_score

    def suggest_next_config(self):
        """Suggest next configuration using balanced acquisition"""
        if len(self.X_evaluated) < 5:
            x = np.random.uniform(0.1, 4.9, len(MODIFIABLE_HEATERS))
            return self.array_to_config(x)
        
        print("  Balanced dual-model candidate selection...")
        
        # Generate diverse candidates
        candidates = []
        
        # 1. Random exploration (30%)
        candidates.extend(np.random.uniform(0.1, 4.9, size=(1500, len(MODIFIABLE_HEATERS))))
        
        # 2. Local search around configs good for AND (25%)
        if self.y_1548_and:
            best_and_indices = np.argsort(self.y_1548_and)[-5:]  # Top 5 AND configs
            for _ in range(1250):
                base_idx = random.choice(best_and_indices)
                base_x = self.X_evaluated[base_idx]
                noise = np.random.normal(0, 0.25, len(MODIFIABLE_HEATERS))
                candidate = np.clip(base_x + noise, 0.1, 4.9)
                candidates.append(candidate)
        
        # 3. Local search around configs good for OR (25%)
        if self.y_1552_or:
            best_or_indices = np.argsort(self.y_1552_or)[-5:]  # Top 5 OR configs
            for _ in range(1250):
                base_idx = random.choice(best_or_indices)
                base_x = self.X_evaluated[base_idx]
                noise = np.random.normal(0, 0.25, len(MODIFIABLE_HEATERS))
                candidate = np.clip(base_x + noise, 0.1, 4.9)
                candidates.append(candidate)
        
        # 4. Local search around balanced configs (20%)
        if len(self.X_evaluated) >= 5:
            # Find configs where both gates scored reasonably
            balanced_indices = []
            for i, (and_score, or_score) in enumerate(zip(self.y_1548_and, self.y_1552_or)):
                if and_score > 0 and or_score > 0:  # Both positive
                    balanced_indices.append(i)
            
            if balanced_indices:
                for _ in range(1000):
                    base_idx = random.choice(balanced_indices)
                    base_x = self.X_evaluated[base_idx]
                    noise = np.random.normal(0, 0.2, len(MODIFIABLE_HEATERS))  # Smaller noise
                    candidate = np.clip(base_x + noise, 0.1, 4.9)
                    candidates.append(candidate)
        
        candidates = np.array(candidates)
        
        # Evaluate balanced acquisition function
        print(f"    Evaluating {len(candidates)} candidates with balanced acquisition...")
        acquisition_values = [self.balanced_acquisition_function(c) for c in candidates]
        
        best_idx = np.argmax(acquisition_values)
        best_candidate = candidates[best_idx]
        best_acq = acquisition_values[best_idx]
        
        print(f"    Best balanced acquisition value: {best_acq:.4f}")
        
        # Show predictions for the selected candidate
        if self.gp_and and self.gp_or:
            x_pred = best_candidate.reshape(1, -1)
            mu_and, sigma_and = self.gp_and.predict(x_pred, return_std=True)
            mu_or, sigma_or = self.gp_or.predict(x_pred, return_std=True)
            print(f"    Predicted AND: {mu_and[0]:.1f} ± {sigma_and[0]:.1f}")
            print(f"    Predicted OR:  {mu_or[0]:.1f} ± {sigma_or[0]:.1f}")
        
        return self.array_to_config(best_candidate)

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
        
        #return min(100, max(-50, total_score))
        return total_score
    
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
            time.sleep(0.25)
            
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
            print(f"  Config {i+1}: {gate_name} score {score:.1f}")
        
        self.turn_laser_off()
        return results

    def evaluate_configs_dual_wavelength(self, configs):
        print(f"Dual-model evaluation of {len(configs)} configurations...")
        
        # Evaluate AND gate at 1548nm
        results_and = self.evaluate_single_wavelength_batch(
            configs, 1548, self.truth_table_1548, self.gate_1548)
        
        # Evaluate OR gate at 1552nm
        results_or = self.evaluate_single_wavelength_batch(
            configs, 1552, self.truth_table_1552, self.gate_1552)
        
        combined_results = []
        for i, config in enumerate(configs):
            _, score_and, details_and = results_and[i]
            _, score_or, details_or = results_or[i]
            
            # Add to dual model datasets
            self.add_evaluation(config, score_and, score_or)
            
            # Calculate combined score for tracking best
            if score_and < -400 or score_or < -400:
                combined_score = -1000
            else:
                # Balanced scoring - both must work
                if score_and > 30 and score_or > 30:
                    # Both working well - multiplicative bonus
                    combined_score = (score_and + score_or) * 1.2
                elif score_and > 0 and score_or > 0:
                    # Both positive but not great
                    combined_score = (score_and + score_or) * 0.8
                elif score_and > 0 or score_or > 0:
                    # Only one working
                    combined_score = max(score_and, score_or) * 0.3
                else:
                    # Both poor
                    combined_score = (score_and + score_or) * 0.1
            
            # Update best configuration tracking
            if combined_score > self.best_combined_score:
                self.best_combined_score = combined_score
                self.best_and_score = score_and
                self.best_or_score = score_or
                self.best_config = config.copy()
                self.best_details = {
                    'and_gate': details_and,
                    'or_gate': details_or,
                    'scores': {'and': score_and, 'or': score_or, 'combined': combined_score}
                }
                print(f"  NEW BEST balanced config: AND={score_and:.1f}, OR={score_or:.1f}, Combined={combined_score:.1f}")
            
            combined_results.append((config, combined_score, {
                'and': (score_and, details_and),
                'or': (score_or, details_or)
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
        print(f"Initial sampling complete. Best AND: {self.best_and_score:.1f}, Best OR: {self.best_or_score:.1f}")

    def bayesian_optimize(self, n_iterations=25, batch_size=5):
        print(f"Dual-model Bayesian optimization: {n_iterations} iterations, batch size {batch_size}")
        
        iteration = 0
        while iteration < n_iterations:
            batch_configs = []
            
            for _ in range(min(batch_size, n_iterations - iteration)):
                config = self.suggest_next_config()
                batch_configs.append(config)
                iteration += 1
            
            self.evaluate_configs_dual_wavelength(batch_configs)
            
            # Fit separate GP models after each batch
            if len(self.X_evaluated) >= 3:
                print("Updating dual GP models...")
                self.fit_gaussian_processes()

    def focused_search_around_best(self, n_variations=12):
        if not self.best_config:
            return
        
        print(f"Focused search around best balanced configuration")
        variations = []
        
        for _ in range(n_variations):
            variation = self.best_config.copy()
            heaters_to_modify = random.sample(MODIFIABLE_HEATERS, 
                                            random.randint(1, min(4, len(MODIFIABLE_HEATERS))))
            
            for h in heaters_to_modify:
                current = variation[h]
                perturbation = random.uniform(-0.15, 0.15)  # Smaller perturbations
                variation[h] = max(0.1, min(4.9, current + perturbation))
            
            variations.append(variation)
        
        self.evaluate_configs_dual_wavelength(variations)

    def test_final_configuration(self):
        if not self.best_config:
            print("No configuration to test.")
            return
        
        print("\nTesting final dual-model configuration")
        print(f"Expected performance - AND: {self.best_and_score:.1f}, OR: {self.best_or_score:.1f}")
        
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
        print(f"\n=== DUAL-MODEL OPTIMIZATION RESULTS ===")
        both_working = self.best_and_score > 30 and self.best_or_score > 30
        if both_working:
            print(f"🎉 SUCCESS: Both gates working! AND={self.best_and_score:.1f}, OR={self.best_or_score:.1f}")
        else:
            print(f"⚠️  Partial success: AND={self.best_and_score:.1f}, OR={self.best_or_score:.1f}")

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
        print("DUAL-MODEL DUAL-WAVELENGTH OPTIMIZATION")
        print("Strategy: Separate models for each gate, balanced acquisition")
        
        self.initial_sampling(n_samples=10)
        self.bayesian_optimize(n_iterations=30, batch_size=8)
        self.focused_search_around_best(n_variations=12)
        self.bayesian_optimize(n_iterations=25, batch_size=6)
        self.focused_search_around_best(n_variations=8)
        self.test_final_configuration()
        
        print(f"Final configuration: {self.format_config()}")
        self.save_results()
        self.cleanup()
        
        return self.best_config, self.best_and_score, self.best_or_score

def main():
    start_time = time.time()
    optimizer = DualModelOptimizer(GATE_1548, GATE_1552)
    
    best_config, best_and, best_or = optimizer.optimize()
    
    if best_config:
        print(f"OPTIMIZATION COMPLETE!")
        print(f"Best AND score: {best_and:.2f}")
        print(f"Best OR score: {best_or:.2f}")
        both_working = best_and > 30 and best_or > 30
        print(f"Both gates working: {'YES' if both_working else 'NO'}")
    else:
        print("OPTIMIZATION FAILED")
    
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main()