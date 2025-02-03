import numpy as np
import serial
import time
import pyvisa
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# =============================
# Configuration
# =============================
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

# =============================
# Coarse Search Parameters
# =============================
NUM_COARSE_SAMPLES = 200             # How many random configurations to test in the coarse phase
COARSE_ALLOWED_VALUES = [0.1, 1.0, 2.0, 3.0, 4.0, 4.9]  # Discrete steps for coarse search
WAIT_TIME = 0.2                    # Wait time (seconds) after setting a heater config
NUM_MEASUREMENTS_PER_STATE = 2       # Number of repeated measurements to average for each state

# =============================
# Local Refinement Parameters
# =============================
LOCAL_REFINEMENT_STEP = 0.3
LOCAL_REFINEMENT_MAX_ITERS = 20  # Limit local search to avoid extremely long loops

class DecoderOptimizer:
    def __init__(self):
        # Initialize hardware resources
        self.scopes = self._init_scopes()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Initialize heater sets
        self.modifiable_heaters = [i for i in range(40) if i not in INPUT_HEATERS]
        
        # Some heaters to keep near 0, if needed
        # (Feel free to remove or tweak if you don't actually need them pinned)
        self.fixed_first_layer = list(range(33, 40))
        
        # Precompute the expected "highest output" channel for each input state
        # If your assignment of channels to the 7 outputs is different, adjust here
        self.expected_outputs = {
            (0.1, 0.1, 0.1): 0,  # 000 -> Channel index 0
            (0.1, 0.1, 4.9): 1,  # 001 -> Channel index 1
            (0.1, 4.9, 0.1): 2,  # 010 -> Channel index 2
            (0.1, 4.9, 4.9): 3,  # 011 -> Channel index 3
            (4.9, 0.1, 0.1): 4,  # 100 -> Channel index 4
            (4.9, 0.1, 4.9): 5,  # 101 -> Channel index 5
            (4.9, 4.9, 0.1): 6   # 110 -> Channel index 6
        }

    def _init_scopes(self):
        rm = pyvisa.ResourceManager()
        SCOPE1_ID = 'USB0::0x1AB1::0x0610::HDO1B244000779::INSTR'
        SCOPE2_ID = 'USB0::0x1AB1::0x0610::HDO1B244100809::INSTR'
        
        scope1 = rm.open_resource(SCOPE1_ID)
        scope2 = rm.open_resource(SCOPE2_ID)
        scopes = [scope1, scope2]
        
        for scope in scopes:
            scope.timeout = 5000
            # Example scope setup - tweak to your actual signals
            for i in range(1, 5):
                scope.write(f':CHANnel{i}:DISPlay ON')
                scope.write(f':CHANnel{i}:SCALe 2')
                scope.write(f':CHANnel{i}:OFFSet -6')
        
        return scopes

    # =================================================================
    # -------------- Measurement and Hardware Communication ------------
    # =================================================================
    def measure_scope(self, scope_idx, output_queue):
        try:
            scope = self.scopes[scope_idx]
            outputs = []
            
            # Scope 1 has 4 channels, scope 2 has 3 channels
            num_channels = 4 if scope_idx == 0 else 3
            
            for channel in range(1, num_channels + 1):
                value = float(scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                outputs.append(round(value, 5))
            
            output_queue.put((scope_idx, outputs))
        except Exception as e:
            print(f"Error measuring scope {scope_idx}: {e}")
            output_queue.put((scope_idx, [None] * (4 if scope_idx == 0 else 3)))

    def measure_outputs_once(self):
        """Measure outputs from both oscilloscopes (one pass). 
           Returns a list of 7 voltages (4 from scope1 + 3 from scope2) or None on error."""
        try:
            output_queue = Queue()
            threads = []
            
            for scope_idx in range(2):
                thread = threading.Thread(target=self.measure_scope, 
                                          args=(scope_idx, output_queue))
                threads.append(thread)
                thread.start()
            
            # Wait for threads to complete
            for thread in threads:
                thread.join()
            
            scope_results = {}
            while not output_queue.empty():
                scope_idx, outputs = output_queue.get()
                scope_results[scope_idx] = outputs
            
            # Combine results in order
            if 0 in scope_results and 1 in scope_results:
                return scope_results[0] + scope_results[1]
            else:
                print("Error: Missing results from one or both scopes.")
                return None
        except Exception as e:
            print(f"Measurement error: {e}")
            return None

    def measure_outputs_average(self, num_samples=3):
        """Measure outputs multiple times and return the average."""
        valid_reads = []
        for _ in range(num_samples):
            result = self.measure_outputs_once()
            if result is not None:
                valid_reads.append(result)
            time.sleep(0.2)  # short delay between repeated measurements
        
        if not valid_reads:
            return None
        
        # Average across all valid reads
        arr = np.array(valid_reads)
        mean_vals = arr.mean(axis=0)
        return list(mean_vals)

    def send_heater_values(self, config):
        message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
        self.serial.write(message.encode())
        self.serial.flush()
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    # =================================================================
    # -------------- Scoring / Fitness  -------------------------------
    # =================================================================
    def evaluate_configuration(self, config):
        """
        Returns a numeric score for a single configuration.
        Higher is better. We'll do a simplified scoring:
          - For each input state, measure outputs
          - The "correct" channel (expected_highest) is rewarded for being high
          - All other channels are rewarded for being low
          - Additional separation bonus
        """
        total_score = 0.0
        
        for input_state in INPUT_STATES:
            # Apply the input state
            test_config = config.copy()
            for i, val in enumerate(input_state):
                test_config[INPUT_HEATERS[i]] = val
            
            # Send to hardware & wait
            self.send_heater_values(test_config)
            time.sleep(WAIT_TIME)
            
            # Measure outputs
            outputs = self.measure_outputs_average(NUM_MEASUREMENTS_PER_STATE)
            if outputs is None or any(x is None for x in outputs):
                # Return a very bad score if measurement fails
                return -1e6
            
            # Identify correct channel
            correct_idx = self.expected_outputs[input_state]
            correct_val = outputs[correct_idx]
            
            # Score each channel
            # Example scoring: 
            #   - correct channel wants to be near 5V => add +some_factor * correct_val
            #   - other channels want to be near 0V => add +(5 - channel_val) for each
            # We'll also add a small separation bonus
            state_score = 0.0
            
            # Reward correct channel for being high
            state_score += correct_val * 30.0  # 30 points per volt for correct channel
            
            # Penalize other channels for being high
            # or equivalently reward them for being low
            for ch_idx, val in enumerate(outputs):
                if ch_idx != correct_idx:
                    state_score += (5.0 - val) * 5.0  # 5 points per volt below 5
            
            # Add a separation ratio bonus
            # correct channel vs. best of the "other" channels
            other_vals = [outputs[i] for i in range(len(outputs)) if i != correct_idx]
            max_other = max(other_vals)
            sep_ratio = (correct_val + 1e-6) / (max_other + 1e-6)
            
            # For example, +20 points if ratio > 2, scaled down otherwise
            if sep_ratio > 2.0:
                state_score += 20.0
            elif sep_ratio > 1.2:
                # partial bonus
                state_score += 20.0 * (sep_ratio - 1.2) / (0.8)  # linear fade
            
            total_score += state_score
        
        return total_score

    # =================================================================
    # -------------- Coarse Random Search  ----------------------------
    # =================================================================
    def create_random_configuration(self, discrete_values=None):
        """
        Create a random configuration, optionally using 
        a discrete set of values for each heater.
        """
        config = {}
        for heater in self.modifiable_heaters:
            if heater in self.fixed_first_layer:
                # Keep "first layer" near 0 V if needed
                config[heater] = 0.01
            else:
                if discrete_values:
                    config[heater] = random.choice(discrete_values)
                else:
                    config[heater] = random.uniform(0.1, 4.9)
        return config

    def coarse_random_search(self, num_samples=200, discrete_values=None):
        """
        Generates `num_samples` random configurations, evaluates them,
        returns a sorted list (descending) of (config, score).
        """
        results = []
        for _ in range(num_samples):
            cfg = self.create_random_configuration(discrete_values)
            score = self.evaluate_configuration(cfg)
            results.append((cfg, score))
        
        # Sort by score, descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # =================================================================
    # -------------- Local Refinement  --------------------------------
    # =================================================================
    def local_refinement(self, config, step=0.3, max_iters=20):
        """
        Simple hill-climbing approach:
          - Evaluate `config`
          - For each heater in modifiable range, try +/- step
          - Keep it if it improves the score
          - Repeat until no improvement or max_iters reached
        """
        best_config = config.copy()
        best_score = self.evaluate_configuration(best_config)
        
        for iteration in range(max_iters):
            improved = False
            for heater in self.modifiable_heaters:
                if heater in self.fixed_first_layer:
                    continue
                current_val = best_config[heater]
                
                for direction in [-1, 1]:
                    trial_config = best_config.copy()
                    new_val = trial_config[heater] + direction * step
                    new_val = max(0.1, min(4.9, new_val))  # clamp
                    trial_config[heater] = new_val
                    
                    trial_score = self.evaluate_configuration(trial_config)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_config = trial_config
                        improved = True
                        # break out of the direction loop to re-check next heater
                        break
                # If we improved, we might want to re-check from the first heater
                if improved:
                    break
            
            if not improved:
                # No improvement in this iteration => we are done
                break
        
        return best_config, best_score

    # =================================================================
    # -------------- Main Optimization Routine -------------------------
    # =================================================================
    def optimize(self):
        """
        Single "good" approach combining:
          1) Coarse random search
          2) Local refinement of the best results
        """
        print(f"\n--- Starting Coarse Random Search with {NUM_COARSE_SAMPLES} samples ---")
        start_time = time.time()
        coarse_results = self.coarse_random_search(
            num_samples=NUM_COARSE_SAMPLES, 
            discrete_values=COARSE_ALLOWED_VALUES
        )
        elapsed_coarse = time.time() - start_time
        print(f"Coarse phase completed in {elapsed_coarse:.2f} seconds.")
        
        # Take top 3 from coarse
        top_candidates = coarse_results[:3]
        
        # Local refinement on each
        best_config = None
        best_score = -1e6
        
        for idx, (cfg, base_score) in enumerate(top_candidates, start=1):
            print(f"\nLocal refinement on candidate #{idx} with coarse score {base_score:.2f} ...")
            refined_cfg, refined_score = self.local_refinement(cfg, step=LOCAL_REFINEMENT_STEP, max_iters=LOCAL_REFINEMENT_MAX_ITERS)
            print(f"Refined score: {refined_score:.2f}")
            
            if refined_score > best_score:
                best_score = refined_score
                best_config = refined_cfg
        
        print("\n--- Optimization Complete ---")
        print(f"Best final score: {best_score:.2f}")
        return best_config, best_score

    def cleanup(self):
        self.serial.close()
        for scope in self.scopes:
            scope.close()


def main():
    optimizer = DecoderOptimizer()
    try:
        best_config, best_score = optimizer.optimize()
        
        # Print best solution
        print("\nBest Heater Configuration found:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f} V")
        
        # Test the best configuration on all input states
        print("\nTesting final configuration on all inputs:")
        for input_state in INPUT_STATES:
            test_cfg = best_config.copy()
            for i, val in enumerate(input_state):
                test_cfg[INPUT_HEATERS[i]] = val
                
            optimizer.send_heater_values(test_cfg)
            time.sleep(WAIT_TIME)
            outputs = optimizer.measure_outputs_average(NUM_MEASUREMENTS_PER_STATE)
            if outputs is None:
                print(f"Measurement failed for input {input_state}")
                continue
            
            max_output = max(outputs)
            max_idx = outputs.index(max_output)
            print(f"\nInput state {input_state}:")
            print(f"Outputs: {['%.3f' % o for o in outputs]}")
            print(f"Highest output channel: {max_idx+1}, Voltage={max_output:.3f}V")
            other_outputs = [o for i, o in enumerate(outputs) if i != max_idx]
            print(f"Separation from next highest: {max_output - max(other_outputs):.3f}V")
            
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    main()
