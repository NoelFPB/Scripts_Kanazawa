import serial
import time
import pyvisa
import json
import random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import numpy as np
# I found the best configuration with this one in the past

# Enhanced Constants
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
CACHE_SIZE = 1024
<<<<<<< HEAD
=======
# Create voltage options from 0.1 to 4.9 with 0.1 step
VOLTAGE_OPTIONS = [round(v/10, 1) for v in range(1, 50)]
# This gives us: [0.1, 0.2, 0.3, ..., 4.8, 4.9]

>>>>>>> 94ca373c3bf1095beacd4b09dde8c22444170aad
INPUT_COMBINATIONS = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
# Define continuous range for voltage
VOLTAGE_MIN = 0.1
VOLTAGE_MAX = 4.9


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

class GeneticOptimizer:
    """Implements genetic algorithm with advanced features"""
    def __init__(self, hardware: HardwareInterface, config_manager: ConfigurationManager):
        self.hardware = hardware
        self.config_manager = config_manager
        self.best_configs_history = []  # Track best configurations
    
    @lru_cache(maxsize=CACHE_SIZE)
        
    def evaluate_single_input(self, config_str: str, input_state: tuple) -> float:
        config = json.loads(config_str)
        config["36"] = input_state[0]
        config["37"] = input_state[1]
        
        self.hardware.send_heater_values(config)
        outputs = self.hardware.measure_outputs()
        
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
            return -10.0  # Penalty for wrong channel
            
        # Base score for correct channel
        score = 10
        
        # Bonus points for separation ONLY if correct channel
        other_outputs = outputs.copy()
        other_outputs.pop(actual_highest)
        separation = max_output - max(other_outputs)
        bonus = min(separation * 5, 15)  # Cap bonus at 10 points
     
        return score + bonus

    def evaluate_configuration(self, config: Dict[str, float]) -> float:
        """Evaluate full configuration"""
        config_str = json.dumps(config, sort_keys=True)
        # Average loss across all input combinations
        return sum(self.evaluate_single_input(config_str, input_state) 
                for input_state in INPUT_COMBINATIONS) / len(INPUT_COMBINATIONS)
    
    def tournament_select(self, population: List[Dict[str, float]], 
                         fitness: List[float], 
                         tournament_size: int = 4) -> Dict[str, float]:
        """Enhanced tournament selection"""
        tournament_idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness[i] for i in tournament_idx]
        winner_idx = tournament_idx[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx].copy()
        
    def adaptive_crossover(self, parent1: Dict[str, float], 
                        parent2: Dict[str, float],
                        generation: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Simple crossover without channel grouping"""
        child1, child2 = {}, {}
        
        # For each heater, randomly choose from either parent
        for h in range(40):  # All heaters 0-39
            if random.random() < 0.5:
                child1[str(h)] = parent1[str(h)]
                child2[str(h)] = parent2[str(h)]
            else:
                child1[str(h)] = parent2[str(h)]
                child2[str(h)] = parent1[str(h)]
        
        return child1, child2
    
<<<<<<< HEAD
    def adaptive_mutation(self, config: Dict[str, float], rate: float, generation: int) -> Dict[str, float]:
=======
    
    def adaptive_mutation(self, config: Dict[str, float], 
                    rate: float, 
                    generation: int) -> Dict[str, float]:
>>>>>>> 94ca373c3bf1095beacd4b09dde8c22444170aad
        result = config.copy()
        temperature = max(1.0 - (generation / 50), 0.1)
        
        for heater in self.config_manager.modifiable_heaters:
            if random.random() < rate * temperature:
                current_value = result[str(heater)]
<<<<<<< HEAD
                
                if random.random() < 0.5:
                    # Global exploration: completely random value
                    result[str(heater)] = round(random.uniform(VOLTAGE_MIN, VOLTAGE_MAX), 3)
                else:
                    # Local refinement: small adjustments
                    delta = random.uniform(-0.1, 0.1)  # Small change
                    new_value = max(VOLTAGE_MIN, min(VOLTAGE_MAX, current_value + delta))
                    result[str(heater)] = round(new_value, 3)
        
        return result
    
    def local_refinement(self, config: Dict[str, float]) -> Dict[str, float]:
        refined_config = config.copy()
        for heater, value in refined_config.items():
            # Small adjustments to improve separation
            best_value = value
            best_score = self.evaluate_configuration(refined_config)
            
            for delta in [-0.05, 0.05]:  # Test small changes
                new_value = max(VOLTAGE_MIN, min(VOLTAGE_MAX, value + delta))
                refined_config[heater] = round(new_value, 3)
                score = self.evaluate_configuration(refined_config)
                if score > best_score:
                    best_value, best_score = new_value, score
            
            refined_config[heater] = best_value
        
        return refined_config


=======
                if random.random() < temperature:
                    # Global exploration: completely random value
                    result[str(heater)] = random.choice(VOLTAGE_OPTIONS)
                else:
                    # Local exploration: move up/down by small steps
                    current_idx = VOLTAGE_OPTIONS.index(current_value)
                    # Number of steps to move scales with temperature
                    max_steps = max(1, int(10 * temperature))  # At most 10 steps (1V) early on
                    step = random.randint(-max_steps, max_steps)
                    new_idx = max(0, min(len(VOLTAGE_OPTIONS)-1, current_idx + step))
                    result[str(heater)] = VOLTAGE_OPTIONS[new_idx]
        
        return result
>>>>>>> 94ca373c3bf1095beacd4b09dde8c22444170aad

    def optimize(self, 
                pop_size: int = 40,
                generations: int = 50,
                base_mutation_rate: float = 0.3,
                early_stop: int = 15) -> Tuple[Dict[str, float], float]:
        """Main optimization function with advanced features"""
        population = [
            self.config_manager.generate_random_config() 
            for _ in range(pop_size)
        ]
        
        best_config = None
        best_score = float('-inf')
        no_improvement_count = 0
        
        for generation in range(generations):
            start_time = time.time()
            
            # Adaptive parameters
            current_mutation_rate = min(
                base_mutation_rate * (1 + no_improvement_count * 0.1),
                0.8
            )
            
            # Evaluate population
            fitness = [self.evaluate_configuration(config) for config in population]
            
            # Population statistics
            avg_fitness = sum(fitness) / len(fitness)
            fitness_std = (sum((f - avg_fitness) ** 2 for f in fitness) / len(fitness)) ** 0.5
            
            # Update best solution
            max_fitness_idx = fitness.index(max(fitness))
            if fitness[max_fitness_idx] > best_score:
                best_score = fitness[max_fitness_idx]
                best_config = population[max_fitness_idx].copy()
                self.best_configs_history.append((best_config, best_score))
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Population restart mechanism
            if no_improvement_count >= early_stop // 3:
                print("Implementing partial population restart...")
                population[pop_size//4:] = [
                    self.config_manager.generate_random_config() 
                    for _ in range(3 * pop_size//4)
                ]
                no_improvement_count = early_stop // 4  # Partial reset of counter
            
            # Early stopping with minimum generations
            if no_improvement_count >= early_stop and generation >= 25:
                print(f"Early stopping at generation {generation + 1}")
                break
            
            # Selection and reproduction
            new_population = [best_config]  # Elitism
            
            # Sort population by fitness
            sorted_indices = sorted(range(len(population)), 
                                 key=lambda i: fitness[i], 
                                 reverse=True)
            top_performers = [population[i] for i in sorted_indices[:pop_size//3]]
            
            while len(new_population) < pop_size:
                if random.random() < 0.7:  # 70% chance of using top performers
                    parent1 = random.choice(top_performers)
                    parent2 = random.choice(top_performers)
                else:  # 30% chance of using tournament selection
                    parent1 = self.tournament_select(population, fitness)
                    parent2 = self.tournament_select(population, fitness)
                
                child1, child2 = self.adaptive_crossover(parent1, parent2, generation)
                
                # Adaptive mutation
                new_population.extend([
                    self.adaptive_mutation(child1, current_mutation_rate, generation),
                    self.adaptive_mutation(child2, current_mutation_rate, generation)
                ])
            
            population = new_population[:pop_size]
            
            gen_time = time.time() - start_time
            print(f"Generation {generation + 1}/{generations}: "
                  f"Best Score = {best_score:.2f}, "
                  f"Avg Score = {avg_fitness:.2f}, "
                  f"Std Dev = {fitness_std:.2f}, "
                  f"Mutation Rate = {current_mutation_rate:.3f}, "
                  f"Time = {gen_time:.2f}s")
        
        return best_config, best_score

def main():
    print("start")
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        hardware = HardwareInterface()
        optimizer = GeneticOptimizer(hardware, config_manager)
        
        # Run optimization
        best_config, best_score = optimizer.optimize(
            pop_size=60,
            generations=55,
            base_mutation_rate=0.1,
            early_stop=35
        )
        
        print("\nOptimization Complete!")
        print(f"Best Score: {best_score}")
        print(f"Best Configuration: {best_config}")
        

            # Print final heater configuration
        print("\nFinal Heater Configuration:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f}V")
        
        # Test final configuration with detailed analysis
        print("\nTesting final configuration:")
        for input_state in INPUT_COMBINATIONS:
            current_config = best_config.copy()
            current_config[36] = input_state[0]
            current_config[37] = input_state[1]
            
            hardware.send_heater_values(current_config)
            time.sleep(0.25)
            outputs = hardware.measure_outputs()
            
            max_output = max(outputs)
            max_index = outputs.index(max_output)
            
            print(f"\nInputs (A, B): {input_state}")
            print(f"Outputs: {outputs}")
            print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
            other_outputs = outputs.copy()
            other_outputs.pop(max_index)
            print(f"Separation from next highest: {(max_output - max(other_outputs)):.4f}V")
    
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        hardware.cleanup()

if __name__ == "__main__":
    main()