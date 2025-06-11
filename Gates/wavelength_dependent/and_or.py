import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import qmc
import json
from datetime import datetime

# Serial port configuration 
SERIAL_PORT = 'COM4'
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
        # Initialize gate types and truth tables
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
            print(f"❌ Wavelength setting failed: {e}")
            return False
    
    def turn_laser_on(self):
        """Turn laser ON"""
        try:
            self.laser.write('LE1')
            time.sleep(1)
            return True
        except Exception as e:
            print(f"❌ Laser ON failed: {e}")
            return False
    
    def turn_laser_off(self):
        """Turn laser OFF"""
        try:
            self.laser.write('LE0')
            time.sleep(1)
            return True
        except Exception as e:
            print(f"❌ Laser OFF failed: {e}")
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
    
    def evaluate_gate_at_wavelength(self, config, wavelength, truth_table, gate_name):
        """Evaluate how well a configuration performs as a specific gate at a wavelength"""
        
        # Set wavelength
        if not self.set_wavelength(wavelength):
            return -1000, {}
        
        if not self.turn_laser_on():
            return -1000, {}
        
        time.sleep(1)  # Let everything settle
        
        # Component weights for single gate evaluation
        WEIGHTS = {
            'high_consistency': 0.2,
            'low_consistency': 0.1,   
            'separation': 0.6,
            'success_bonus': 0.1
        }
        
        component_scores = {k: 0.0 for k in WEIGHTS}
        
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
            time.sleep(0.25)  # Longer settling time for wavelength changes
            
            output = self.measure_output()
            if output is None:
                self.turn_laser_off()
                return -1000, {}
            
            detailed_results[input_state] = {
                'output': output,
                'expected_high': expected_high
            }
            
            if expected_high:
                high_outputs.append(output)
            else:
                low_outputs.append(output)
        
        self.turn_laser_off()
        
        # If missing states, penalize
        if not high_outputs or not low_outputs:
            return -500, detailed_results
        
        # Calculate consistency scores
        if len(high_outputs) > 1:
            avg_high = sum(high_outputs) / len(high_outputs)
            high_variance = sum((o - avg_high)**2 for o in high_outputs) / len(high_outputs)
            component_scores['high_consistency'] = 1.0 / (1.0 + (high_variance / 0.25))
        else:
            component_scores['high_consistency'] = 1.0
        
        if len(low_outputs) > 1:
            avg_low = sum(low_outputs) / len(low_outputs)
            low_variance = sum((o - avg_low)**2 for o in low_outputs) / len(low_outputs)
            component_scores['low_consistency'] = 1.0 / (1.0 + (low_variance / 0.25))
        else:
            component_scores['low_consistency'] = 1.0
        
        # Separation score
        avg_high = sum(high_outputs) / len(high_outputs)
        avg_low = sum(low_outputs) / len(low_outputs)
        separation = avg_high - avg_low
        
        all_outputs = high_outputs + low_outputs
        voltage_range_used = max(all_outputs) - min(all_outputs)
        
        if voltage_range_used > 0:
            normalized_separation = abs(separation) / voltage_range_used
            component_scores['separation'] = min(1.0, normalized_separation * 2)
        else:
            component_scores['separation'] = 0.0
        
        # Success bonus
        success_count = 0
        for high_out in high_outputs:
            for low_out in low_outputs:
                if high_out > low_out:
                    success_count += 1
        
        total_comparisons = len(high_outputs) * len(low_outputs)
        success_rate = success_count / total_comparisons if total_comparisons > 0 else 0
        component_scores['success_bonus'] = success_rate
        
        # Final score
        final_score = sum(component_scores[k] * WEIGHTS[k] * 100 for k in WEIGHTS)
        
        # Add detailed analysis
        detailed_results['summary'] = {
            'gate_name': gate_name,
            'wavelength': wavelength,
            'avg_high': avg_high,
            'avg_low': avg_low,
            'separation': separation,
            'success_rate': success_rate,
            'final_score': final_score,
            'component_scores': component_scores
        }
        
        return final_score, detailed_results
    
    def evaluate_single_wavelength_batch(self, configs, wavelength, truth_table, gate_name):
        """Evaluate multiple configurations at a single wavelength (batch mode)"""
        print(f"\nSetting laser to {wavelength}nm for {gate_name} gate testing...")
        
        # Set wavelength once
        if not self.set_wavelength(wavelength):
            return [(config, -1000, {}) for config in configs]
        
        if not self.turn_laser_on():
            return [(config, -1000, {}) for config in configs]
        
        print(f"Waiting 25 seconds for laser to stabilize...")
        time.sleep(25)  # Wait for laser to reach operating condition
        
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
        """Evaluate a single configuration at the current wavelength (laser already stable)"""
        
        # Component weights for single gate evaluation
        WEIGHTS = {
            'high_consistency': 0.2,
            'low_consistency': 0.1,   
            'separation': 0.6,
            'success_bonus': 0.1
        }
        
        component_scores = {k: 0.0 for k in WEIGHTS}
        
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
        
        # Calculate consistency scores
        if len(high_outputs) > 1:
            avg_high = sum(high_outputs) / len(high_outputs)
            high_variance = sum((o - avg_high)**2 for o in high_outputs) / len(high_outputs)
            component_scores['high_consistency'] = 1.0 / (1.0 + (high_variance / 0.25))
        else:
            component_scores['high_consistency'] = 1.0
        
        if len(low_outputs) > 1:
            avg_low = sum(low_outputs) / len(low_outputs)
            low_variance = sum((o - avg_low)**2 for o in low_outputs) / len(low_outputs)
            component_scores['low_consistency'] = 1.0 / (1.0 + (low_variance / 0.25))
        else:
            component_scores['low_consistency'] = 1.0
        
        # Separation score
        avg_high = sum(high_outputs) / len(high_outputs)
        avg_low = sum(low_outputs) / len(low_outputs)
        separation = avg_high - avg_low
        
        all_outputs = high_outputs + low_outputs
        voltage_range_used = max(all_outputs) - min(all_outputs)
        
        if voltage_range_used > 0:
            normalized_separation = abs(separation) / voltage_range_used
            component_scores['separation'] = min(1.0, normalized_separation * 2)
        else:
            component_scores['separation'] = 0.0
        
        # Success bonus
        success_count = 0
        for high_out in high_outputs:
            for low_out in low_outputs:
                if high_out > low_out:
                    success_count += 1
        
        total_comparisons = len(high_outputs) * len(low_outputs)
        success_rate = success_count / total_comparisons if total_comparisons > 0 else 0
        component_scores['success_bonus'] = success_rate
        
        # Final score
        final_score = sum(component_scores[k] * WEIGHTS[k] * 100 for k in WEIGHTS)
        
        # Add detailed analysis
        detailed_results['summary'] = {
            'gate_name': gate_name,
            'wavelength': wavelength,
            'avg_high': avg_high,
            'avg_low': avg_low,
            'separation': separation,
            'success_rate': success_rate,
            'final_score': final_score,
            'component_scores': component_scores
        }
        
        return final_score, detailed_results
    
    def evaluate_configs_dual_wavelength(self, configs):
        """Evaluate multiple configurations at both wavelengths using batch approach"""
        print(f"\nDUAL-WAVELENGTH BATCH EVALUATION")
        print(f"Evaluating {len(configs)} configurations at both wavelengths...")
        
        # Batch evaluate at 1548nm
        results_1548 = self.evaluate_single_wavelength_batch(
            configs, 1548, self.truth_table_1548, self.gate_1548)
        
        time.sleep(2)  # Brief pause between wavelengths
        
        # Batch evaluate at 1552nm
        results_1552 = self.evaluate_single_wavelength_batch(
            configs, 1552, self.truth_table_1552, self.gate_1552)
        
        # Combine results
        combined_results = []
        for i, config in enumerate(configs):
            _, score_1548, details_1548 = results_1548[i]
            _, score_1552, details_1552 = results_1552[i]
            
            # Combined scoring
            if score_1548 < -500 or score_1552 < -500:
                combined_score = -2000
            else:
                min_score = min(score_1548, score_1552)
                avg_score = (score_1548 + score_1552) / 2
                
                if min_score < 20:
                    combined_score = min_score * 0.5
                else:
                    combined_score = avg_score + (min_score * 0.3)
            
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
                        'combined': combined_score,
                        'min_score': min_score,
                        'avg_score': avg_score
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
        """SPSA optimization using batch evaluation"""
        
        if not self.best_config:
            print("❌ No initial configuration available for SPSA")
            return
        
        theta = {h: self.best_config.get(h, 0.1) for h in MODIFIABLE_HEATERS}
        heater_keys = sorted(MODIFIABLE_HEATERS)
        iterations_without_improvement = 0
        
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
            
            # Process results and update theta
            best_iteration_score = -float('inf')
            best_iteration_theta = None
            
            for i, (theta_plus, theta_minus, delta) in enumerate(configs_to_test):
                # Get scores for this perturbation pair
                idx_plus = i * 2
                idx_minus = i * 2 + 1
                
                _, score_plus, _ = results[idx_plus]
                _, score_minus, _ = results[idx_minus]
                
                # Estimate gradient for this perturbation
                g_hat = {h: (score_plus - score_minus) / (2 * ck * delta[h]) for h in heater_keys}
                
                # Create new theta candidate
                theta_candidate = {h: max(0.1, min(4.9, theta[h] + ak * g_hat[h])) for h in heater_keys}
                
                # Track best candidate from this iteration
                candidate_score = max(score_plus, score_minus)
                if candidate_score > best_iteration_score:
                    best_iteration_score = candidate_score
                    if score_plus > score_minus:
                        best_iteration_theta = configs_to_test[i][0].copy()  # theta_plus
                    else:
                        best_iteration_theta = configs_to_test[i][1].copy()  # theta_minus
            
            # Update theta based on best result from this iteration
            if best_iteration_score > self.best_score:
                theta = best_iteration_theta.copy()
                iterations_without_improvement = 0
                print(f"Improved! New best score: {self.best_score:.1f}")
            else:
                iterations_without_improvement += 1
                
                if iterations_without_improvement >= 3:  # Restart sooner with batch mode
                    print(f"No improvement for {iterations_without_improvement} iterations. Restarting from best")
                    theta = {h: self.best_config.get(h, 0.1) for h in MODIFIABLE_HEATERS}
                    iterations_without_improvement = 0
                else:
                    # Small random perturbation to continue exploring
                    theta = {h: max(0.1, min(4.9, v + random.uniform(-0.1, 0.1))) 
                            for h, v in theta.items()}
            
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
        time.sleep(1)
        
        for input_state in INPUT_COMBINATIONS:
            current_config = self.best_config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.3)
            output = self.measure_output()
            expected = self.truth_table_1548[input_state]
            
            print(f"  Inputs {input_state}: {output:.4f}V (expect {'HIGH' if expected else 'LOW'})")
        
        self.turn_laser_off()
        time.sleep(1)
        
        # Test at 1552nm (OR gate)
        print(f"\nTesting at 1552nm ({self.gate_1552} gate):")
        self.set_wavelength(1552)
        self.turn_laser_on()
        time.sleep(1)
        
        for input_state in INPUT_COMBINATIONS:
            current_config = self.best_config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.3)
            output = self.measure_output()
            expected = self.truth_table_1552[input_state]
            
            print(f"  Inputs {input_state}: {output:.4f}V (expect {'HIGH' if expected else 'LOW'})")
        
        self.turn_laser_off()
        
        # Print best details if available
        if self.best_details:
            print(f"\nPERFORMANCE SUMMARY:")
            scores = self.best_details['scores']
            print(f"  1548nm {self.gate_1548} score: {scores['1548nm']:.1f}/100")
            print(f"  1552nm {self.gate_1552} score: {scores['1552nm']:.1f}/100")
            print(f"  Combined score: {scores['combined']:.1f}")
    
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
        """Run the full dual-wavelength optimization with batch approach"""
        print(f"\nDUAL-WAVELENGTH BATCH OPTIMIZATION STARTING")
        print(f"Target: {self.gate_1548} at 1548nm, {self.gate_1552} at 1552nm")
        print(f"Laser settling time: 25 seconds per wavelength")
        print("=" * 60)
        
        try:
            # Phase 1: Initial sampling (batch mode)
            print(f"\nPHASE 1: Initial Sampling")
            self.initial_sampling(n_samples=20)
            
            # Phase 2: SPSA optimization (batch mode)
            print(f"\nPHASE 2: SPSA Optimization")
            self.spsa_batch_optimize(iterations=8, batch_size=6)
            
            # Phase 3: Focused search around best
            print(f"\nPHASE 3: Focused Search")
            self.focused_search_around_best(n_variations=12)
            
            # Phase 4: Final testing
            print(f"\nPHASE 4: Final Validation")
            self.test_final_configuration()
            
            print(f"\nFINAL CONFIGURATION:")
            config = self.format_config()
            print(json.dumps(config, indent=2))
            
            # Calculate total time estimate
            total_wavelength_switches = 2 * (20 + 8*6 + 12 + 2)  # Rough estimate
            estimated_time = total_wavelength_switches * 25 / 60  # Minutes
            print(f"\nEstimated total optimization time: ~{estimated_time:.0f} minutes")
            
            self.save_results()
            
            return self.best_config, self.best_score
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None, -1
        
        finally:
            self.cleanup()

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("🌈 DUAL-WAVELENGTH LOGIC GATE OPTIMIZER")
    print("Searching for heater configuration that produces:")
    print("  - AND gate behavior at 1548nm")
    print("  - OR gate behavior at 1552nm")
    print()
    
    optimizer = DualWavelengthOptimizer(GATE_1548, GATE_1552)
    
    try:
        best_config, best_score = optimizer.optimize()
        
        if best_config:
            print(f"\n🎉 OPTIMIZATION COMPLETE!")
            print(f"Best combined score: {best_score:.2f}")
        else:
            print(f"\n❌ OPTIMIZATION FAILED")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Optimization interrupted by user")
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        
    finally:
        try:
            optimizer.cleanup()
        except:
            pass
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n⏱️ Total execution time: {execution_time/60:.1f} minutes")

if __name__ == "__main__":
    main()