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

# --- HELPER FUNCTION ---
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
        self.X_evaluated, self.y_1548, self.y_1552, self.y_combined = [], [], [], []
        self.gp_1548, self.gp_1552, self.gp_combined = None, None, None
        self.scaler = StandardScaler()
        
        # Best configurations tracking
        self.best_config, self.best_1548_only, self.best_1552_only = None, None, None
        self.best_score, self.best_1548_score, self.best_1552_score = float('-inf'), float('-inf'), float('-inf')
        self.best_details = {}
        self.evaluation_history = []
        
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
        scope.write(':CHANnel2:DISPlay ON'); scope.write(':CHANnel2:SCALe 2'); scope.write(':CHANnel2:OFFSet -6')
        return scope
    
    def _init_laser(self):
        rm = pyvisa.ResourceManager()
        laser = rm.open_resource(LASER_ADDRESS)
        laser.timeout = 5000; laser.write_termination = ''; laser.read_termination = ''
        return laser

    def config_to_array(self, config):
        return np.array([config[h] for h in MODIFIABLE_HEATERS])

    def array_to_config(self, x):
        return {h: x[i] for i, h in enumerate(MODIFIABLE_HEATERS)}

    def add_evaluation(self, config, score_1548, score_1552, score_combined, details=None):
        self.X_evaluated.append(self.config_to_array(config))
        self.y_1548.append(score_1548)
        self.y_1552.append(score_1552)
        self.y_combined.append(score_combined)
        
        self.evaluation_history.append({
            'config': config.copy(),
            'scores': {'1548nm': score_1548, '1552nm': score_1552, 'combined': score_combined},
            'details': details
        })

    def fit_gaussian_processes(self):
        if len(self.X_evaluated) < 5: return
        X = np.array(self.X_evaluated)
        try:
            X_scaled = self.scaler.transform(X)
        except:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
        
        kernel = (ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1))
        
        self.gp_1548 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42)
        self.gp_1552 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=43)
        self.gp_combined = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=44)
        
        self.gp_1548.fit(X_scaled, np.array(self.y_1548))
        self.gp_1552.fit(X_scaled, np.array(self.y_1552))
        self.gp_combined.fit(X_scaled, np.array(self.y_combined))

    def acquisition_function(self, x, mode='combined', beta=2.0):
        if self.gp_combined is None: return random.random()
        x = x.reshape(1, -1)
        try:
            x_scaled = self.scaler.transform(x)
        except:
            return random.random()
        
        if mode == 'combined':
            mu, sigma = self.gp_combined.predict(x_scaled, return_std=True)
        elif mode == '1548':
            mu, sigma = self.gp_1548.predict(x_scaled, return_std=True)
        else: # '1552'
            mu, sigma = self.gp_1552.predict(x_scaled, return_std=True)
            
        return mu[0] + beta * sigma[0]


    def suggest_next_configs(self, n_configs=5, focused_bounds_center=None):
            if len(self.X_evaluated) < 10 and focused_bounds_center is None:
                return [self.array_to_config(np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))) for _ in range(n_configs)]
            
            if focused_bounds_center:
                # Create narrow bounds around the provided center point
                new_bounds = []
                radius = 0.5 # Search within +/- 0.5V
                for heater in MODIFIABLE_HEATERS:
                    center_val = focused_bounds_center.get(heater, 2.5)
                    new_bounds.append((max(V_MIN, center_val - radius), min(V_MAX, center_val + radius)))
                bounds = new_bounds
            else:
                # Use the full, wide bounds for a fresh run
                bounds = [(V_MIN, V_MAX)] * len(MODIFIABLE_HEATERS)
            
            configs = []
            def obj_func(x, mode):
                return -self.acquisition_function(x, mode=mode, beta=2.5)

            for i in range(n_configs):
                mode = ['combined', '1548', '1552'][i % 3]
                result = differential_evolution(lambda x: obj_func(x, mode), bounds, maxiter=50, popsize=15, polish=True)
                configs.append(self.array_to_config(result.x))
            return configs

    def calculate_extinction_ratio_score(self, high_outputs, low_outputs):
        if not high_outputs or not low_outputs:
            return -2000.0  # Severe penalty for no outputs

        min_high = min(high_outputs)
        max_low = max(low_outputs)
        separation = min_high - max_low

        # If logic levels overlap, this is a complete failure. Punish severely.
        if separation <= 0:
            return -1000.0 + 100.0 * separation # Penalty proportional to the overlap

        # --- Tiered Scoring Logic ---
        # We now score based on achieving certain performance tiers.

        er_db = 10 * np.log10(min_high / max(max_low, 1e-3))
        
        # Tier 1: Is the ER even usable? (e.g., > 3 dB)
        if er_db < 3.0:
            # If not, the score is low and is ONLY based on ER. Don't even consider consistency yet.
            # This forces the optimizer to find a decent ER first.
            return er_db * 10 - 50 # A score between -20 and -50 for 0-3dB ER

        # Tier 2: The ER is acceptable. NOW we can factor in consistency.
        # Start with a base score for achieving a good ER
        score = 50 + (er_db - 3.0) * 15 # e.g., 50 points for 3dB, 80 points for 5dB

        # Add a balanced penalty for inconsistency
        penalty_factor = 75.0 

        if len(low_outputs) > 1:
            low_std_dev = np.std(low_outputs)
            # The penalty is now smaller relative to the base score
            score -= penalty_factor * low_std_dev

        if len(high_outputs) > 1:
            high_std_dev = np.std(high_outputs)
            score -= penalty_factor * high_std_dev

        return score


    def _calculate_combined_score(self, score_1548, score_1552):
        """Helper to compute combined score, rewarding balance."""
        if score_1548 < 0 and score_1552 < 0:
            return score_1548 + score_1552 # Sum of penalties
            
        # Use geometric mean for positive scores to favor balance
        # Add a small epsilon to handle zero scores
        s1 = max(score_1548, 1e-9)
        s2 = max(score_1552, 1e-9)
        
        if score_1548 > 0 and score_1552 > 0:
             # High reward for both being good
            return np.sqrt(s1 * s2) * 1.5
        else:
            # Penalize if one is good and the other is not
            return min(s1, s2) * 0.5

    def send_heater_values(self, config):
        voltage_message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
        self.serial.write(voltage_message.encode())
        self.serial.flush()
        time.sleep(0.01)

    def measure_output(self):
        return round(float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel2')), 5)

    # --- MISSING LASER CONTROL METHODS ---
    # These methods were accidentally removed during refactoring and are now restored.
    def set_wavelength(self, wavelength_nm):
        self.laser.write(f'LW{wavelength_nm}nm')
        time.sleep(0.5)

    def turn_laser_on(self):
        self.laser.write('LE1')
        time.sleep(0.5)

    def turn_laser_off(self):
        self.laser.write('LE0')
        time.sleep(0.5)
    # ------------------------------------

    def evaluate_single_config_at_wavelength(self, config, truth_table):
        high_outputs, low_outputs = [], []
        detailed_results = {}
        
        for input_state, expected_high in truth_table.items():
            current_config = {**config, **self.base_config}
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.2)
            
            measurements = [self.measure_output() for _ in range(3)] # Reduced measurements for speed
            output = np.median(measurements)
            
            detailed_results[input_state] = {'output': output, 'expected_high': expected_high}
            (high_outputs if expected_high else low_outputs).append(output)
            
        score = self.calculate_extinction_ratio_score(high_outputs, low_outputs)
        return score, detailed_results
        
    def _run_evaluation_at_wavelength(self, configs, wavelength, truth_table, gate_name):
        """Helper to run all evaluations for a given wavelength."""
        print(f"\nSetting laser to {wavelength}nm for {gate_name} gate testing...")
        self.set_wavelength(wavelength)
        self.turn_laser_on()
        time.sleep(12) # Reduced stabilization time

        results = {}
        for i, config in enumerate(configs):
            score, details = self.evaluate_single_config_at_wavelength(config, truth_table)
            results[i] = (score, details)
            print(f"  Config {i+1}/{len(configs)} @ {wavelength}nm: Score {score:.1f}")
        
        self.turn_laser_off()
        print(f"  {wavelength}nm evaluation complete.")
        return results

    def run_multi_phase_evaluation(self, phases):
        """
        OPTIMIZED: Processes configurations from multiple phases in a way that
        minimizes laser wavelength switching.
        """
        all_configs = []
        phase_indices = []
        for name, config_generator in phases.items():
            print(f"\nGenerating configs for phase: {name}")
            new_configs = config_generator()
            start_index = len(all_configs)
            all_configs.extend(new_configs)
            end_index = len(all_configs)
            phase_indices.append({'name': name, 'indices': range(start_index, end_index)})
            print(f"-> Generated {len(new_configs)} configs.")

        if not all_configs:
            print("No configurations generated to evaluate.")
            return

        # --- Bulk Evaluation at 1548nm ---
        results_1548 = self._run_evaluation_at_wavelength(
            all_configs, 1548, self.truth_table_1548, self.gate_1548
        )
        
        # --- Bulk Evaluation at 1552nm ---
        results_1552 = self._run_evaluation_at_wavelength(
            all_configs, 1552, self.truth_table_1552, self.gate_1552
        )

        # --- Process all results and update models/best scores ---
        print("\nProcessing all evaluation results...")
        for i, config in enumerate(all_configs):
            score_1548, details_1548 = results_1548[i]
            score_1552, details_1552 = results_1552[i]
            
            # Update best single-wavelength scores
            if score_1548 > self.best_1548_score:
                self.best_1548_score, self.best_1548_only = score_1548, config.copy()
            if score_1552 > self.best_1552_score:
                self.best_1552_score, self.best_1552_only = score_1552, config.copy()

            combined_score = self._calculate_combined_score(score_1548, score_1552)
            self.add_evaluation(config, score_1548, score_1552, combined_score, {
                '1548nm': details_1548, '1552nm': details_1552
            })
            
            if combined_score > self.best_score:
                self.best_score = combined_score
                self.best_config = config.copy()
                self.best_details = {
                    '1548nm': details_1548, '1552nm': details_1552,
                    'scores': {'1548nm': score_1548, '1552nm': score_1552, 'combined': combined_score}
                }
                print(f"  *** NEW BEST (Combined Score: {combined_score:.1f}) ***")
        
        # --- Fit GP Models after a round of evaluations ---
        print("\nUpdating Gaussian Process models...")
        self.fit_gaussian_processes()


    def test_final_configuration(self):
        # This function remains largely the same, but with updated sleep/measurement counts
        if not self.best_config:
            print("No configuration to test.")
            return
        
        print("\n" + "="*60 + "\nFINAL CONFIGURATION TESTING\n" + "="*60)
        for wavelength, gate, truth_table in [(1548, self.gate_1548, self.truth_table_1548), 
                                              (1552, self.gate_1552, self.truth_table_1552)]:
            print(f"\nTesting at {wavelength}nm ({gate} gate):")
            self.set_wavelength(wavelength)
            self.turn_laser_on()
            time.sleep(12)
            
            high_outputs, low_outputs = [], []
            for input_state, expected in truth_table.items():
                current_config = self.best_config.copy()
                current_config[INPUT_HEATERS[0]], current_config[INPUT_HEATERS[1]] = input_state
                self.send_heater_values(current_config)
                time.sleep(0.3) # Longer sleep for final verification
                
                measurements = [self.measure_output() for _ in range(5)] # More measurements for final test
                output, std = np.median(measurements), np.std(measurements)
                (high_outputs if expected else low_outputs).append(output)
                print(f"  {input_state}: {output:.4f}V ±{std:.4f} ({'HIGH' if expected else 'LOW'})")

            if high_outputs and low_outputs:
                min_high, max_low = min(high_outputs), max(low_outputs)
                if min_high > max_low:
                    er_db = 10 * np.log10(min_high / max(max_low, 0.001))
                    print(f"  -> Performance: ER = {er_db:.2f} dB, Separation = {min_high - max_low:.3f}V")
                else:
                    print(f"  -> WARNING: Overlapping logic levels by {max_low - min_high:.3f}V")
            self.turn_laser_off()

    def save_results(self):
        # This function remains the same
        if not self.best_config: return
        # ... (code omitted for brevity, no changes needed)
        pass

    def cleanup(self):
        self.laser.write('LE0')
        self.serial.close()
        print("Cleanup complete")

    def format_config(self):
        """Creates a complete, formatted dictionary for all 40 heaters from the best config."""
        if not self.best_config:
            return {}
        
        # Start with a base of 0.0 for all heaters
        complete_config = {i: 0.0 for i in range(40)}

        # Add the fixed values for the first layer
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                complete_config[h] = 0.01

        # Overwrite with the optimized values for the modifiable heaters
        for heater, value in self.best_config.items():
            complete_config[heater] = value
        
        # Return a clean, rounded dictionary
        return {k: round(float(v), 4) for k, v in complete_config.items()}

    def save_results(self):
        """Saves the final results and configuration to a JSON file."""
        if not self.best_config:
            print("No best configuration found to save.")
            return

        final_config = self.format_config()
        results = {
            'timestamp': datetime.now().isoformat(),
            'gate_1548': self.gate_1548,
            'gate_1552': self.gate_1552,
            'best_combined_score': float(self.best_score),
            'final_test_results': self.best_details.get('scores', {}),
            'best_config': final_config
        }
        
        filename = f"optimization_results_{self.gate_1548}_{self.gate_1552}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nSuccessfully saved results to {filename}")
        except Exception as e:
            print(f"\n--- ERROR: Could not save results to file: {e} ---")
            print("--- Final Configuration (copy from here) ---")
            print(json.dumps(final_config, indent=4))

    def optimize(self, focused_run=False, prev_best_config=None):
            """
            Main optimization loop.
            - If focused_run is True, it will skip initial sampling and use the 
            prev_best_config to seed the search.
            """
            run_type = "FOCUSED RUN" if focused_run else "FRESH RUN"
            print("\n" + "="*60 + f"\nIMPROVED DUAL-WAVELENGTH OPTIMIZATION ({run_type})\n" + "="*60)
            start_time = time.time()

            if focused_run:
                if prev_best_config is None:
                    print("ERROR: A focused run requires a prev_best_config dictionary.")
                    return None, None # Exit safely
                # --- WARM START FOR FOCUSED RUN ---
                print("\n" + "="*60 + "\nWARM START: Seeding optimizer with known best configuration.\n" + "="*60)
                self.run_multi_phase_evaluation({
                    "Warm Start Seed": lambda: [prev_best_config]
                })
            else:
                # --- PHASE 1: INITIAL EXPLORATION (for a fresh run) ---
                self.run_multi_phase_evaluation({
                    "Initial LHS Sampling": lambda: [
                        self.array_to_config(V_MIN + qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS)).random(n=1)[0] * (V_MAX - V_MIN))
                        for _ in range(40)
                    ]
                })

            # --- PHASE 2: ITERATIVE BAYESIAN OPTIMIZATION ---
            print("\n" + "="*60 + "\nStarting Bayesian Optimization Phase\n" + "="*60)
            stagnation_counter = 0
            last_best_score = self.best_score
            for i in range(15): # Increased iterations
                print(f"\nBayesian Iteration {i+1}/15")
                if self.best_score > last_best_score:
                    print("  -> Score has improved. Resetting stagnation counter.")
                    stagnation_counter = 0
                    last_best_score = self.best_score
                else:
                    stagnation_counter += 1
                    print(f"  -> Score has not improved. Stagnation counter: {stagnation_counter}")

                if stagnation_counter >= 14: # If stuck for 4 iterations...
                    print("  -> STAGNATION DETECTED! Forcing a wide, random search to find new areas.")
                    self.run_multi_phase_evaluation({
                        f"Stagnation Breaker {i+1}": lambda: [self.array_to_config(np.random.uniform(V_MIN, V_MAX, len(MODIFIABLE_HEATERS))) for _ in range(5)]
                    })
                    stagnation_counter = 0 # Reset counter after the shake-up
                else:
                    self.run_multi_phase_evaluation({
                        f"Bayesian Batch {i+1}": lambda: self.suggest_next_configs(n_configs=8, focused_bounds_center=prev_best_config if focused_run else None)
                    })

            # --- PHASE 3: FINAL REFINEMENT ---
            if self.best_config:
                print("\n" + "="*60 + "\nStarting Final Refinement Phase\n" + "="*60)
                self.run_multi_phase_evaluation({
                    "Final Perturbations": lambda: [ self.array_to_config(np.clip(self.config_to_array(self.best_config) + np.random.normal(0, 0.1, len(MODIFIABLE_HEATERS)), V_MIN, V_MAX)) for _ in range(20) ]
                })

            # --- FINAL STEPS ---
            self.test_final_configuration()
            elapsed_time = (time.time() - start_time) / 60
            
            print(f"\n" + "="*60 + f"\nOPTIMIZATION COMPLETE in {elapsed_time:.1f} minutes\n" + "="*60)
            print(f"Total evaluations: {len(self.evaluation_history)}")
            print(f"Best combined score: {self.best_score:.2f}")
            self.save_results()
            self.cleanup()
            
            # This return statement is now correctly placed at the end of the function's scope
            return self.best_config, self.best_score

def main():
    optimizer = ImprovedDualWavelengthOptimizer(GATE_1548, GATE_1552)

    # --- CHOOSE YOUR RUN TYPE ---
    # To run a fresh optimization from scratch:
    # best_config, best_score = optimizer.optimize(focused_run=False)

    # To run a focused run based on a previous best result:
# This dictionary goes inside the main() function
    prev_best_config = {
        0: 1.026,   1: 0.144,   2: 0.886,   3: 1.008,   4: 1.748,
        5: 0.369,   6: 0.488,   7: 4.343,   8: 1.665,   9: 0.852,
        10: 4.412,  11: 4.9,    12: 3.827,  13: 2.578,  14: 1.424,
        15: 1.181,  16: 1.756,  17: 1.307,  18: 3.349,  19: 1.891,
        20: 1.247,  21: 1.641,  22: 0.99,   23: 0.574,  24: 2.944,
        25: 0.117,  26: 2.767,  27: 3.091,  28: 1.281,  29: 2.654,
        30: 4.889,  31: 1.292,  32: 0.421
    }
    best_config, best_score = optimizer.optimize(focused_run=True, prev_best_config=prev_best_config)
    # --------------------------

    print("\n" + "="*60)
    print("                FINAL OPTIMIZATION RESULT")
    print("="*60)
    if best_config:
        final_config_dict = optimizer.format_config()
        print(f"Best Combined Score: {best_score:.2f}")
        print("\nFinal Heater Configuration:")
        print(json.dumps(final_config_dict, indent=4))
    else:
        print("\nOPTIMIZATION FAILED TO FIND A SUITABLE CONFIGURATION.")
    print("="*60)

if __name__ == "__main__":
    main()
