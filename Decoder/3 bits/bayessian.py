import numpy as np
import serial
import time
import pyvisa
import GPy
import GPyOpt
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
INPUT_HEATERS = [35, 36, 37]  # Three input heaters for 3-bit decoder
INPUT_STATES = [
    (0.1, 0.1, 0.1),  # 000
    (0.1, 0.1, 4.9),  # 001
    (0.1, 4.9, 0.1),  # 010
    (0.1, 4.9, 4.9),  # 011
    (4.9, 0.1, 0.1),  # 100
    (4.9, 0.1, 4.9),  # 101
    (4.9, 4.9, 0.1)   # 110
]

class BayesianDecoderOptimizer:
    def __init__(self):
        self.scopes = self._init_scopes()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Initialize heater configuration
        self.modifiable_heaters = [i for i in range(40) if i not in INPUT_HEATERS]
        self.fixed_first_layer = list(range(33, 40))
        
        # Store best configuration and performance
        self.best_config = None
        self.best_score = float('-inf')
        self.evaluation_count = 0
        
        # Initialize optimization history
        self.history = {
            'configs': [],
            'scores': [],
            'best_scores': []
        }
        
    def _init_scopes(self):
        """Initialize oscilloscopes"""
        rm = pyvisa.ResourceManager()
        SCOPE1_ID = 'USB0::0x1AB1::0x0610::HDO1B244000779::INSTR'
        SCOPE2_ID = 'USB0::0x1AB1::0x0610::HDO1B244100809::INSTR'
        
        scope1 = rm.open_resource(SCOPE1_ID)
        scope2 = rm.open_resource(SCOPE2_ID)
        scopes = [scope1, scope2]
        
        for scope in scopes:
            scope.timeout = 5000
            for i in range(1, 5):
                scope.write(f':CHANnel{i}:DISPlay ON')
                scope.write(f':CHANnel{i}:SCALe 2')
                scope.write(f':CHANnel{i}:OFFSet -6')
        
        return scopes

    def measure_scope(self, scope_idx, output_queue):
        """Measure outputs from a single scope"""
        try:
            scope = self.scopes[scope_idx]
            outputs = []
            
            num_channels = 4 if scope_idx == 0 else 3
            
            for channel in range(1, num_channels + 1):
                value = float(scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                outputs.append(round(value, 5))
            
            output_queue.put((scope_idx, outputs))
        except Exception as e:
            print(f"Error measuring scope {scope_idx}: {e}")
            output_queue.put((scope_idx, [None] * (4 if scope_idx == 0 else 3)))

    def measure_outputs(self):
        """Measure outputs from both oscilloscopes in parallel"""
        try:
            output_queue = Queue()
            threads = []
            
            for scope_idx in range(2):
                thread = threading.Thread(target=self.measure_scope, 
                                       args=(scope_idx, output_queue))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            all_outputs = []
            scope_results = {}
            
            while not output_queue.empty():
                scope_idx, outputs = output_queue.get()
                scope_results[scope_idx] = outputs
            
            if 0 in scope_results and 1 in scope_results:
                all_outputs.extend(scope_results[0])
                all_outputs.extend(scope_results[1])
                return all_outputs
            else:
                print("Error: Missing results from one or both scopes")
                return [None] * 7
                
        except Exception as e:
            print(f"Measurement error: {e}")
            return [None] * 7

    def send_heater_values(self, config):
        """Send heater values via serial connection"""
        message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
        self.serial.write(message.encode())
        self.serial.flush()
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def evaluate_configuration(self, config_array):
        """Evaluate a single configuration"""
        # Convert array to dictionary format
        config = {}
        config_idx = 0
        for heater in self.modifiable_heaters:
            if heater in self.fixed_first_layer:
                config[heater] = 0.01
            else:
                config[heater] = float(config_array[0][config_idx])
                config_idx += 1

        total_score = 0
        min_separation = float('inf')
        all_outputs = []

        # Expected output mapping
        expected_outputs = {
            (0.1, 0.1, 0.1): 0,  # 000
            (0.1, 0.1, 4.9): 1,  # 001
            (0.1, 4.9, 0.1): 2,  # 010
            (0.1, 4.9, 4.9): 3,  # 011
            (4.9, 0.1, 0.1): 4,  # 100
            (4.9, 0.1, 4.9): 5,  # 101
            (4.9, 4.9, 0.1): 6   # 110
        }

        for input_state in INPUT_STATES:
            # Apply input state
            current_config = config.copy()
            for i, value in enumerate(input_state):
                current_config[INPUT_HEATERS[i]] = value
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            outputs = self.measure_outputs()
            
            if None in outputs:
                return -1000.0  # Penalty for measurement failure
            
            all_outputs.append(outputs)
            max_output = max(outputs)
            actual_highest = outputs.index(max_output)
            expected_highest = expected_outputs[input_state]
            
            # Calculate separation metrics
            other_outputs = outputs.copy()
            other_outputs.pop(expected_highest)
            max_other = max(other_outputs)
            separation_ratio = outputs[expected_highest] / (max_other + 1e-6)
            min_separation = min(min_separation, separation_ratio)
            
            # Scoring
            if actual_highest == expected_highest:
                # Base score for correct output
                state_score = 150
                
                # Separation quality bonus
                if separation_ratio > 2.0:
                    state_score += 100
                elif separation_ratio > 1.5:
                    state_score += 75
                else:
                    state_score += max(0, (separation_ratio - 1) * 60)
                
                # Voltage level bonus
                if outputs[expected_highest] > 2.0:
                    state_score += 50
                elif outputs[expected_highest] > 1.0:
                    state_score += 30
                else:
                    state_score += max(0, outputs[expected_highest] * 25)
                
                # Extra bonus for very clear separation
                if separation_ratio > 2.5:
                    state_score += 25
            else:
                # Penalty for incorrect output
                voltage_diff = outputs[expected_highest] - max_output
                state_score = -75 + max(0, (voltage_diff + 0.5) * 20)
            
            total_score += state_score
        
        # Update best configuration if necessary
        if total_score > self.best_score:
            self.best_score = total_score
            self.best_config = config.copy()
            print(f"\nNew best score: {total_score:.2f}")
            
        # Update history
        self.history['configs'].append(config)
        self.history['scores'].append(total_score)
        self.history['best_scores'].append(self.best_score)
        
        self.evaluation_count += 1
        
        return float(total_score)

    def optimize(self, max_iter=100):
        """Run Bayesian optimization"""
        # Define optimization space
        num_params = len(self.modifiable_heaters) - len(self.fixed_first_layer)
        bounds = [{'name': f'heater_{i}', 'type': 'continuous', 'domain': (0.1, 4.9)} 
                 for i in range(num_params)]
        
        # Initialize optimizer
        optimizer = GPyOpt.methods.BayesianOptimization(
            f=self.evaluate_configuration,
            domain=bounds,
            model_type='GP',
            acquisition_type='EI',
            normalize_Y=True,
            initial_design_numdata=20,
            exact_feval=False,
            maximize=True
        )
        
        print("\nStarting Bayesian Optimization...")
        print("=" * 100)
        print(f"{'Iter':>4} {'Score':>10} {'Best':>10} {'Time (s)':>10}")
        print("-" * 100)
        
        try:
            start_time = time.time()
            
            # Run optimization
            optimizer.run_optimization(max_iter)
            
            # Get best result
            best_x = optimizer.x_opt
            best_score = optimizer.fx_opt
            
            # Convert best parameters to config dictionary
            final_config = {}
            param_idx = 0
            for heater in self.modifiable_heaters:
                if heater in self.fixed_first_layer:
                    final_config[heater] = 0.01
                else:
                    final_config[heater] = float(best_x[param_idx])
                    param_idx += 1
            
            total_time = (time.time() - start_time) / 60.0
            print("\nOptimization completed!")
            print(f"Best score: {best_score:.2f}")
            print(f"Total evaluations: {self.evaluation_count}")
            print(f"Total time: {total_time:.2f} minutes")
            print("=" * 100)
            
            return final_config, best_score
            
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
            return self.best_config, self.best_score
        
    def test_configuration(self, config):
        """Test a configuration with all input states"""
        print("\nTesting configuration...")
        print("=" * 60)
        
        for input_state in INPUT_STATES:
            current_config = config.copy()
            for i, value in enumerate(input_state):
                current_config[INPUT_HEATERS[i]] = value
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            outputs = self.measure_outputs()
            
            max_output = max(outputs)
            max_index = outputs.index(max_output)
            
            other_outputs = outputs.copy()
            other_outputs.pop(max_index)
            separation = max_output - max(other_outputs)
            
            print(f"\nInputs (A, B, C): {input_state}")
            print(f"Outputs: {[f'{x:.4f}' for x in outputs]}")
            print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
            print(f"Separation from next highest: {separation:.4f}V")
        
    def cleanup(self):
        """Clean up resources"""
        self.serial.close()
        for scope in self.scopes:
            scope.close()

def main():
    optimizer = BayesianDecoderOptimizer()
    try:
        print("Starting 3-bit decoder optimization using Bayesian Optimization...")
        best_config, best_score = optimizer.optimize(max_iter=150)
        
        print("\nBest Configuration Found:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f}V")
        
        # Test the best configuration
        optimizer.test_configuration(best_config)
        
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    main()