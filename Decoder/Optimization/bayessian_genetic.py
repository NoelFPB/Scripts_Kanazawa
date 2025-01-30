from scipy.optimize import minimize
import random
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from typing import Dict, Tuple, List
import json
import time
from typing import Dict, List, Tuple, Optional
import pyvisa
import serial


# Constants
VOLTAGE_MIN = 0.1
VOLTAGE_MAX = 4.9
NUM_HEATERS = 40
GENERATION_LIMIT = 25
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
CACHE_SIZE = 1024
INPUT_COMBINATIONS = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]

class ConfigurationManager:
    """Manages configuration generation and validation"""
    def __init__(self):
        self.modifiable_heaters = sorted([i for i in range(33) if i not in [36, 37]], reverse=True)
        self.fixed_first_layer = list(range(33, 40))
    
    def generate_random_config(self) -> Dict[str, float]:
        """Generate a random configuration with continuous voltage values."""
        config = {
            str(h): round(random.uniform(VOLTAGE_MIN, VOLTAGE_MAX), 3) 
            for h in self.modifiable_heaters
        }
        config.update({
            str(h): round(random.uniform(0.01, 0.5), 3)
            for h in self.fixed_first_layer
        })
        return config


class HardwareInterface:
    """Manages hardware communication"""
    def __init__(self):
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        self.scope = rm.open_resource(resources[0])
        self.scope.timeout = 5000
        
        self._setup_channels()

    def evaluate_single_input(self, config_str: str, input_state: tuple) -> float:
        config = json.loads(config_str)
        config["36"] = input_state[0]
        config["37"] = input_state[1]
        
        self.send_heater_values(config)
        outputs = self.measure_outputs()
        
        if None in outputs:
            return -10.0
        
        # Get target channel and highest output
        target_idx = {
            (0.1, 0.1): 0,
            (0.1, 4.9): 1,
            (4.9, 0.1): 2,  # Channel 3 for high-low
            (4.9, 4.9): 3   # Channel 4 for high-high
        }[input_state]
        
        max_output = max(outputs)
        actual_highest = outputs.index(max_output)
        
        # If wrong channel is highest, return negative score
        if actual_highest != target_idx:
            return -20.0  # Penalty for wrong channel
            
        # Base score for correct channel
        score = 10
        
        # Bonus points for separation ONLY if correct channel
        other_outputs = outputs.copy()
        other_outputs.pop(actual_highest)
        separation = max_output - max(other_outputs)
        bonus = min(separation * 5, 15)  # Cap bonus at 10 points
        return score + bonus

    
    def _setup_channels(self):
        """Configure oscilloscope channels"""
        for channel in ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']:
            self.scope.write(f':{channel}:DISPlay ON')
            self.scope.write(f':{channel}:SCALe 2')
            self.scope.write(f':{channel}:OFFSet -6')
    
    def send_heater_values(self, config: Dict[str, float]) -> None:
        """Send configuration to hardware"""
        voltage_message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
        self.ser.write(voltage_message.encode())
        self.ser.flush()
        time.sleep(0.25)  # Reduced delay 
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
    
    def measure_outputs(self) -> List[Optional[float]]:
        """Measure all channels with error handling"""
        try:
            outputs = []
            for channel in range(1, 5):
                value = float(self.scope.query(
                    f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'
                ))
                outputs.append(round(value, 5))
            return outputs
        except Exception as e:
            print(f"Measurement error: {e}")
            return [None] * 4
    
    def cleanup(self):
        """Clean up hardware connections"""
        self.ser.close()
        self.scope.close()


# Helper: Convert configuration dict to list and back
def config_to_list(config: Dict[str, float]) -> List[float]:
    return [config[str(h)] for h in range(NUM_HEATERS)]

def list_to_config(lst: List[float]) -> Dict[str, float]:
    return {str(h): round(lst[h], 3) for h in range(len(lst))}

class HybridOptimizer:
    def __init__(self, hardware, config_manager):
        self.hardware = hardware
        self.config_manager = config_manager
        self.best_config = None
        self.best_score = float('-inf')

    def genetic_algorithm(self, pop_size: int, generations: int) -> List[Dict[str, float]]:
        population = [
            self.config_manager.generate_random_config() 
            for _ in range(pop_size)
        ]
        
        for generation in range(generations):
            print(f"Generation {generation}")
            fitness = [self.evaluate_configuration(config) for config in population]
            
            # Add diversity check
            unique_configs = set(json.dumps(config, sort_keys=True) for config in population)
            if len(unique_configs) < pop_size * 0.5:  # If population diversity is too low
                # Inject new random individuals
                num_new = int(pop_size * 0.2)
                population[-num_new:] = [
                    self.config_manager.generate_random_config() 
                    for _ in range(num_new)
                ]
                # Update best configuration
            max_idx = np.argmax(fitness)
            if fitness[max_idx] > self.best_score:
                self.best_score = fitness[max_idx]
                self.best_config = population[max_idx]
                print(f"New best score found: {self.best_score:.2f}")  # Added best score notification
        
            # Select parents
            top_performers = [
                population[i] for i in np.argsort(fitness)[-pop_size//2:]
            ]
            
            # Generate next generation
            new_population = []
            while len(new_population) < pop_size:
                parent1, parent2 = random.sample(top_performers, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([
                    self.mutate(child1, generation),
                    self.mutate(child2, generation)
                ])
            population = new_population[:pop_size]
        
        return top_performers  # Return best solutions for Bayesian optimization

    def evaluate_configuration(self, config: Dict[str, float]) -> float:
        config_str = json.dumps(config, sort_keys=True)
        scores = [self.hardware.evaluate_single_input(config_str, input_state) 
                for input_state in INPUT_COMBINATIONS]
        average_score = sum(scores) / len(scores)
        print(f"Configuration evaluation results:")
        for input_state, score in zip(INPUT_COMBINATIONS, scores):
            print(f"  Input {input_state}: {score:.2f}")
        print(f"  Average score: {average_score:.2f}")
        return average_score

    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        child1, child2 = {}, {}
        for h in range(NUM_HEATERS):
            if random.random() < 0.5:
                child1[str(h)] = parent1[str(h)]
                child2[str(h)] = parent2[str(h)]
            else:
                child1[str(h)] = parent2[str(h)]
                child2[str(h)] = parent1[str(h)]
        return child1, child2
    
    def mutate(self, config: Dict[str, float], generation: int) -> Dict[str, float]:
        result = config.copy()
        temperature = max(1.0 - (generation / GENERATION_LIMIT), 0.3)  # Higher minimum temperature
        base_mutation_rate = 0.2  # Increased from 0.1
        for h in range(NUM_HEATERS):
            if random.random() < base_mutation_rate * temperature:
                # Add Gaussian noise instead of uniform random
                current_value = result[str(h)]
                noise = random.gauss(0, temperature * (VOLTAGE_MAX - VOLTAGE_MIN) / 4)
                new_value = max(VOLTAGE_MIN, min(VOLTAGE_MAX, current_value + noise))
                result[str(h)] = round(new_value, 3)
        return result

    def bayesian_optimization(self, initial_population: List[Dict[str, float]]) -> Tuple[Dict[str, float], float]:
        print(f"\nStarting Bayesian Optimization with {len(initial_population)} initial points")    
        space = [Real(VOLTAGE_MIN, VOLTAGE_MAX, name=f"h{i}") for i in range(NUM_HEATERS)]
        iteration_count = 0
        best_score_so_far = float('-inf')
        best_config_so_far = None
        
        # Keep track of good configurations and their scores
        good_configs = []
        good_scores = []
        
        @use_named_args(space)
        def objective(**config_list):
            nonlocal iteration_count, best_score_so_far, best_config_so_far
            iteration_count += 1
            print(f"\nBayesian Optimization Iteration {iteration_count}/50")
            
            config = list_to_config([config_list[f"h{i}"] for i in range(NUM_HEATERS)])
            score = self.evaluate_configuration(config)
            
            # Store good configurations (those with positive scores)
            if score > 0:
                good_configs.append(config)
                good_scores.append(score)
            
            # Update best seen configuration
            if score > best_score_so_far:
                best_score_so_far = score
                best_config_so_far = config
                print(f"New best score: {score}")
            
            return -score  # Negative for minimization

        # Convert initial population to list format
        initial_points = [config_to_list(config) for config in initial_population]

        # Run Bayesian Optimization with enhanced parameters
        res = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=50,
            n_initial_points=len(initial_points),
            x0=initial_points,
            # Added parameters for better exploration/exploitation balance
            noise=0.1,  # Account for measurement noise
            n_random_starts=5,  # More random exploration initially
            acq_func='EI',  # Expected Improvement acquisition function
            acq_func_kwargs={'xi': 0.1},  # Decreased exploration tendency
            n_jobs=1,  # Ensure sequential evaluation
            callback=lambda res: print(f"Current minimum: {-res.fun}"),
        )

        # If the final result is worse than our best seen configuration,
        # return the best seen instead
        final_config = list_to_config(res.x)
        final_score = -res.fun
        
        if best_score_so_far > final_score:
            return best_config_so_far, best_score_so_far
        return final_config, final_score

    # Hybrid Optimization
    def optimize(self, pop_size: int, ga_generations: int) -> Tuple[Dict[str, float], float]:
        # Step 1: Genetic Algorithm
        print("Running Genetic Algorithm...")
        top_performers = self.genetic_algorithm(pop_size, ga_generations)

        # Step 2: Bayesian Optimization
        print("Running Bayesian Optimization...")
        refined_config, refined_score = self.bayesian_optimization(top_performers)

        # Update final results
        if refined_score > self.best_score:
            self.best_score = refined_score
            self.best_config = refined_config

        return self.best_config, self.best_score


# Main Function
def main():

    print("Starting Hybrid Optimization...")
    config_manager = ConfigurationManager()
    hardware = HardwareInterface()
    optimizer = HybridOptimizer(hardware, config_manager)

    best_config, best_score = optimizer.optimize(pop_size=20, ga_generations=45)

    print("\nOptimization Complete!")
    print(f"Best Score: {best_score}")
    print(f"Best Configuration: {best_config}")



if __name__ == "__main__":
    main()
