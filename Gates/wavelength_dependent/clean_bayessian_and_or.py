import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc
import json
from datetime import datetime
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Hardware configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
LASER_ADDRESS = "GPIB0::6::INSTR"

# Gate configuration
GATE_1548 = "AND"
GATE_1552 = "OR"
INPUT_HEATERS = [36, 37]

# Voltage definitions
V_MIN = 0.1
V_MAX = 4.9

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

class DualWavelengthOptimizer:
    def __init__(self, gate_1548=GATE_1548, gate_1552=GATE_1552):
        self.gate_1548 = gate_1548
        self.gate_1552 = gate_1552
        self.truth_table_1548 = generate_truth_table(gate_1548)
        self.truth_table_1552 = generate_truth_table(gate_1552)
        
        # Bayesian optimization storage
        self.X_evaluated = []
        self.y_combined = []
        self.gp_combined = None
        
        # Hardware connections
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        self.laser = self._init_laser()
        time.sleep(1)
        
        # Best configuration tracking
        self.best_config = None
        self.best_score = float('-inf')
        
        print(f"Optimizing dual-wavelength logic gates:")
        print(f"  {gate_1548} gate at 1548nm")
        print(f"  {gate_1552} gate at 1552nm")

    def _init_scope(self):
        """Initialize oscilloscope"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No scope VISA resources found")
        
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
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

    def config_to_array(self, config):
        """Convert config dict to numpy array"""
        return np.array([config[h] for h in MODIFIABLE_HEATERS])

    def array_to_config(self, x):
        """Convert numpy array to config dict"""
        return {h: x[i] for i, h in enumerate(MODIFIABLE_HEATERS)}

    def set_wavelength(self, wavelength_nm):
        """Set laser wavelength"""
        try:
            self.laser.write(f'LW{wavelength_nm}nm')
            time.sleep(2)
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
        """Send heater voltage configuration to hardware with retry logic"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        
        # Add retry logic for serial communication
        for attempt in range(3):
            try:
                self.serial.write(voltage_message.encode())
                self.serial.flush()
                time.sleep(0.02)  # Slightly longer delay
                self.serial.reset_input_buffer()
                self.serial.reset_output_buffer()
                return True
            except Exception as e:
                print(f"Serial communication attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(0.1)
                    continue
                return False
        return False

    def measure_output(self):
        """Measure the logic gate output voltage from oscilloscope"""
        try:
            # Add retry logic for flaky measurements
            for attempt in range(3):
                try:
                    value = float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel2'))
                    if value is not None and not math.isnan(value):
                        return round(value, 5)
                except:
                    if attempt < 2:  # Don't sleep on last attempt
                        time.sleep(0.1)
                        continue
                    raise
            
            print(f"Measurement returned invalid value after 3 attempts")
            return None
            
        except Exception as e:
            print(f"Measurement error: {e}")
            return None

    def calculate_score(self, high_outputs, low_outputs):
        """
        Bayesian-friendly scoring with informative gradients even for poor configurations
        
        Key insight: Instead of harsh penalties, provide gradual feedback that helps
        the optimizer learn which direction to move to improve separation
        """
        if not high_outputs or not low_outputs:
            return -50
        
        min_high = min(high_outputs)
        max_low = max(low_outputs)
        mean_high = sum(high_outputs) / len(high_outputs)
        mean_low = sum(low_outputs) / len(low_outputs)
        
        # === SEPARATION ANALYSIS ===
        separation = min_high - max_low
        
        # === INFORMATIVE OVERLAP HANDLING ===
        # Instead of harsh -1000, provide gradual feedback based on HOW BAD the overlap is
        if separation <= 0:
            # Measure the "quality" of the outputs even when overlapping
            high_quality = mean_high / 5.0  # How high are the intended HIGHs? (0-1 scale)
            low_quality = max(0, 1 - mean_low / 3.0)  # How low are the intended LOWs? (0-1 scale)
            
            # Overlap severity (how much do they overlap?)
            overlap_severity = abs(separation) / 2.0  # Normalize overlap amount
            
            # Give partial credit based on output quality, penalize overlap
            base_score = (high_quality + low_quality) * 20  # 0-40 points for output quality
            overlap_penalty = min(50, overlap_severity * 25)  # 0-50 penalty for overlap
            
            overlap_score = base_score - overlap_penalty - 20  # Shift to negative range
            return max(-100, overlap_score)  # Cap at -100, not -1000
        
        # === WORKING CONFIGURATION SCORING ===
        # For non-overlapping configurations, use the original sophisticated scoring
        
        # 1. Extinction Ratio (50 points)
        extinction_ratio = mean_high / max(mean_low, 0.001)
        er_score = 50 * (1 - math.exp(-extinction_ratio / 10))
        
        # 2. Separation Robustness (30 points)
        separation_normalized = separation / 1.0
        separation_score = 30 * math.tanh(separation_normalized * 2)
        
        # 3. Output Quality (15 points)
        max_output = max(max(high_outputs), max(low_outputs))
        quality_score = 15 / (1 + math.exp(-2 * (max_output - 2)))
        
        # 4. Consistency Bonus (5 points)
        high_std = np.std(high_outputs) if len(high_outputs) > 1 else 0
        low_std = np.std(low_outputs) if len(low_outputs) > 1 else 0
        avg_std = (high_std + low_std) / 2
        consistency_score = 5 * math.exp(-avg_std * 10)
        
        total_score = er_score + separation_score + quality_score + consistency_score
        
        # Smooth clipping
        if total_score < 0:
            final_score = -10 / (1 + math.exp(-total_score / 20))
        elif total_score > 100:
            final_score = 100 - 10 / (1 + math.exp((total_score - 100) / 20))
        else:
            final_score = total_score
        
        return final_score

    def evaluate_at_wavelength(self, config, wavelength, truth_table):
        """Evaluate a single configuration at given wavelength"""
        high_outputs = []
        low_outputs = []
        failed_measurements = 0
        
        # Check if heater communication works first
        if not self.send_heater_values(config):
            print(f"    Heater communication failed")
            return -1000
        
        for i, input_state in enumerate(INPUT_COMBINATIONS):
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            expected_high = truth_table[input_state]
            
            # Send heater values with error checking
            if not self.send_heater_values(current_config):
                failed_measurements += 1
                print(f"    Heater setting {i+1}/4 failed for input {input_state}")
                continue
                
            time.sleep(0.4)  # Longer settling time
            
            output = self.measure_output()
            if output is None:
                failed_measurements += 1
                print(f"    Measurement {i+1}/4 failed for input {input_state}")
                continue
            
            if expected_high:
                high_outputs.append(output)
            else:
                low_outputs.append(output)
        
        # If too many measurements failed, return failure
        if failed_measurements >= 3:
            print(f"    Too many failures ({failed_measurements}/4)")
            return -1000
        
        # If we don't have both high and low outputs, return partial failure
        if not high_outputs or not low_outputs:
            print(f"    Missing output types: high={len(high_outputs)}, low={len(low_outputs)}")
            return -500
        
        return self.calculate_score(high_outputs, low_outputs)

    def evaluate_batch_at_wavelength(self, configs, wavelength, truth_table, gate_name):
        """Evaluate multiple configurations at a single wavelength"""
        print(f"Setting laser to {wavelength}nm for {gate_name} gate testing...")
        
        if not self.set_wavelength(wavelength):
            print(f"ERROR: Failed to set wavelength to {wavelength}nm")
            return [-1000] * len(configs)
            
        if not self.turn_laser_on():
            print(f"ERROR: Failed to turn laser ON")
            return [-1000] * len(configs)
        
        print(f"Waiting for laser to stabilize at {wavelength}nm...")
        time.sleep(14)  # Laser stabilization
        
        # Test if scope is responding
        try:
            test_measurement = self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel2')
            print(f"Scope test measurement: {test_measurement}")
        except Exception as e:
            print(f"ERROR: Scope not responding: {e}")
            self.turn_laser_off()
            return [-1000] * len(configs)
        
        scores = []
        successful_measurements = 0
        
        for i, config in enumerate(configs):
            score = self.evaluate_at_wavelength(config, wavelength, truth_table)
            scores.append(score)
            
            if score > -500:  # Any successful measurement
                successful_measurements += 1
                
            # Show more detail for failed measurements
            if score >= 50:
                print(f"  Config {i+1}/{len(configs)}: {score:.1f} (excellent)")
            elif score >= 0:
                print(f"  Config {i+1}/{len(configs)}: {score:.1f} (working)")
            elif score >= -50:
                print(f"  Config {i+1}/{len(configs)}: {score:.1f} (near miss)")
            elif score >= -100:
                print(f"  Config {i+1}/{len(configs)}: {score:.1f} (poor overlap)")
            else:
                print(f"  Config {i+1}/{len(configs)}: {score:.1f} (failed)")
        
        print(f"Successful measurements: {successful_measurements}/{len(configs)}")
        
        self.turn_laser_off()
        return scores

    def evaluate_configs_dual_wavelength(self, configs):
        """Evaluate configurations at both wavelengths"""
        print(f"Evaluating {len(configs)} configurations at both wavelengths...")
        
        # Evaluate at both wavelengths
        scores_1548 = self.evaluate_batch_at_wavelength(
            configs, 1548, self.truth_table_1548, self.gate_1548)
        time.sleep(1)
        scores_1552 = self.evaluate_batch_at_wavelength(
            configs, 1552, self.truth_table_1552, self.gate_1552)
        
        # Combine results
        results = []
        for i, config in enumerate(configs):
            score_1548 = scores_1548[i]
            score_1552 = scores_1552[i]
            
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
                print(f"  NEW BEST: 1548nm={score_1548:.1f}, 1552nm={score_1552:.1f}, Combined={combined_score:.1f}")
            
            results.append((config, combined_score))
            
            # Add to Bayesian optimization data
            self.X_evaluated.append(self.config_to_array(config))
            self.y_combined.append(combined_score)
        
        return results

    def fit_gaussian_process(self):
        """Fit Gaussian Process with better handling of extreme values"""
        if len(self.X_evaluated) < 5:
            return
        
        X = np.array(self.X_evaluated)
        y = np.array(self.y_combined)
        
        # Filter out extreme failures for better GP fitting
        # Keep some failures for learning, but not all -1000s
        mask = y > -800  # Keep partial failures (-500) but filter most -1000s
        if np.sum(mask) < len(y) * 0.3:  # Keep at least 30% of data
            # If too few good points, keep best 50% of data
            threshold = np.percentile(y, 50)
            mask = y >= threshold
        
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        print(f"    Fitting GP with {len(X_filtered)}/{len(X)} points (filtered extreme failures)")
        print(f"    Score range: [{y_filtered.min():.1f}, {y_filtered.max():.1f}]")
        
        # Better kernel bounds to avoid convergence warnings
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.01, 1000)) *  # Wider bounds
            RBF(length_scale=1.0, length_scale_bounds=(0.01, 100.0)) +
            WhiteKernel(noise_level=1.0, noise_level_bounds=(0.001, 100.0))  # Wider bounds
        )
        
        try:
            self.gp_combined = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=15,  # More restarts
                alpha=1e-6,
                normalize_y=True,  # Normalize for better numerical stability
                random_state=42
            )
            
            self.gp_combined.fit(X_filtered, y_filtered)
            print("    GP fitted successfully (filtered dataset)")
            
        except Exception as e:
            print(f"    GP fitting failed: {e}")
            self.gp_combined = None

    def suggest_next_config(self):
        """Suggest next configuration using Bayesian optimization"""
        if self.gp_combined is None or len(self.X_evaluated) < 5:
            # Random exploration
            x = np.random.uniform(0.1, 4.9, len(MODIFIABLE_HEATERS))
            return self.array_to_config(x)
        
        # Generate candidates
        n_candidates = 5000
        candidates = np.random.uniform(0.1, 4.9, size=(n_candidates, len(MODIFIABLE_HEATERS)))
        
        # UCB acquisition function
        mu, sigma = self.gp_combined.predict(candidates, return_std=True)
        beta = 5.0
        ucb = mu + beta * sigma
        
        # Select best candidate
        best_idx = np.argmax(ucb)
        return self.array_to_config(candidates[best_idx])

    def initial_sampling(self, n_samples=20):
        """Smarter initial sampling biased toward working configurations"""
        print(f"Initial sampling - {n_samples} configurations")
        
        configs = []
        
        # Start with some known-good patterns (lower voltages tend to work better)
        good_patterns = [
            {h: 0.5 for h in MODIFIABLE_HEATERS},  # Low uniform
            {h: 1.0 for h in MODIFIABLE_HEATERS},  # Medium uniform
            {h: 0.1 if i % 2 == 0 else 1.5 for i, h in enumerate(MODIFIABLE_HEATERS)},  # Alternating
            {h: 0.1 + (i % 4) * 0.5 for i, h in enumerate(MODIFIABLE_HEATERS)},  # Stepped pattern
        ]
        
        # Add some good patterns
        for pattern in good_patterns[:min(4, n_samples//4)]:
            configs.append(pattern)
        
        # Biased random sampling for the rest
        sampler = qmc.LatinHypercube(d=len(MODIFIABLE_HEATERS), seed=42)
        n_remaining = n_samples - len(configs)
        samples = sampler.random(n=n_remaining)
        
        for sample in samples:
            config = {}
            for j, h in enumerate(MODIFIABLE_HEATERS):
                # Bias toward lower voltages (0.1-2.5V range instead of 0.1-4.9V)
                # This reduces the chance of overlapping logic levels
                biased_range = 2.4  # Instead of 4.8
                config[h] = V_MIN + sample[j] * biased_range
            configs.append(config)
        
        self.evaluate_configs_dual_wavelength(configs)
        print(f"Initial sampling complete. Best score: {self.best_score:.1f}")
        
        # Print success rate for diagnostics
        success_rate = len([x for x in self.y_combined if x > -500]) / len(self.y_combined)
        print(f"Success rate: {success_rate:.1%} ({len([x for x in self.y_combined if x > -500])}/{len(self.y_combined)} configs worked)")
        
        return len([x for x in self.y_combined if x > -500])  # Return number of successful configs

    def bayesian_optimize(self, n_iterations=25, batch_size=5):
        """Bayesian optimization with adaptive batching"""
        print(f"Bayesian optimization - {n_iterations} iterations, initial batch size {batch_size}")
        
        iteration = 0
        current_batch_size = batch_size
        
        while iteration < n_iterations:
            # Adaptive batch sizing based on recent success rate
            if len(self.y_combined) > 10:
                recent_scores = self.y_combined[-10:]  # Last 10 evaluations
                success_rate = len([x for x in recent_scores if x > -500]) / len(recent_scores)
                
                if success_rate < 0.2:  # Less than 20% success
                    current_batch_size = max(2, current_batch_size - 1)
                    print(f"  Low success rate ({success_rate:.1%}), reducing batch size to {current_batch_size}")
                elif success_rate > 0.6:  # More than 60% success
                    current_batch_size = min(8, current_batch_size + 1)
                    print(f"  High success rate ({success_rate:.1%}), increasing batch size to {current_batch_size}")
            
            # Generate batch of configurations
            batch_configs = []
            for _ in range(min(current_batch_size, n_iterations - iteration)):
                config = self.suggest_next_config()
                batch_configs.append(config)
                iteration += 1
            
            # Evaluate batch
            self.evaluate_configs_dual_wavelength(batch_configs)
            
            # Update GP
            self.fit_gaussian_process()
        
        total_success = len([x for x in self.y_combined if x > -500])
        print(f"Bayesian optimization complete. Best score: {self.best_score:.1f}")
        print(f"Overall success rate: {total_success/len(self.y_combined):.1%} ({total_success}/{len(self.y_combined)})")

    def test_final_configuration(self):
        """Test the final configuration at both wavelengths"""
        if not self.best_config:
            print("No configuration to test.")
            return
        
        print("TESTING FINAL DUAL-WAVELENGTH CONFIGURATION")
        print("=" * 60)
        
        for wavelength, gate_name, truth_table in [
            (1548, self.gate_1548, self.truth_table_1548),
            (1552, self.gate_1552, self.truth_table_1552)
        ]:
            print(f"\nTesting at {wavelength}nm ({gate_name} gate):")
            
            self.set_wavelength(wavelength)
            self.turn_laser_on()
            time.sleep(14)
            
            outputs = []
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
                
                outputs.append(output)
                if expected:
                    high_outputs.append(output)
                else:
                    low_outputs.append(output)
                
                print(f"  Inputs {input_state}: {output:.4f}V (expect {'HIGH' if expected else 'LOW'})")
            
            # Calculate metrics
            if high_outputs and low_outputs:
                mean_high = sum(high_outputs) / len(high_outputs)
                mean_low = sum(low_outputs) / len(low_outputs)
                separation = min(high_outputs) - max(low_outputs)
                er_db = 10 * math.log10(mean_high / max(mean_low, 0.001))
                
                print(f"  --> Extinction Ratio: {er_db:.2f} dB")
                print(f"  --> Worst-case separation: {separation:.3f}V")
                print(f"  --> HIGH levels: {mean_high:.3f}V")
                print(f"  --> LOW levels: {mean_low:.3f}V")
            
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
        """Save results to JSON file"""
        if not self.best_config:
            return
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'gate_1548': self.gate_1548,
            'gate_1552': self.gate_1552,
            'best_score': float(self.best_score),
            'best_config': self.format_config()
        }
        
        filename = f"dual_wavelength_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
            print("Final configuration:")
            print(json.dumps(self.format_config(), indent=2))

    def cleanup(self):
        """Clean up hardware connections"""
        print("Cleaning up connections...")
        
        for device_name, device in [('laser', self.laser), ('serial', self.serial), ('scope', self.scope)]:
            try:
                if device_name == 'laser' and device:
                    device.write('LE0')  # Turn off laser
                    time.sleep(0.5)
                if device:
                    device.close()
                print(f"{device_name.capitalize()} connection closed")
            except Exception as e:
                print(f"Warning: Could not close {device_name}: {e}")

    def optimize(self):
        """Main optimization routine"""
        print("DUAL-WAVELENGTH BATCH OPTIMIZATION STARTING")
        print(f"Target: {self.gate_1548} at 1548nm, {self.gate_1552} at 1552nm")
        print("=" * 60)
        
        try:
            # Phase 1: Initial sampling
            self.initial_sampling(n_samples=30)
            
            # Phase 2: Bayesian optimization
            self.bayesian_optimize(n_iterations=30, batch_size=10)
            
            # Phase 3: Final testing
            self.test_final_configuration()
            
            print("\nFINAL CONFIGURATION:")
            print(self.format_config())
            
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