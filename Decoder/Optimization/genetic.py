import serial
import time
import pyvisa
import json
import random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
# I found the best configuration with this one in the past

# Enhanced Constants
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
CACHE_SIZE = 1024
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
        time.sleep(0.2)  # Reduced delay 
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
        
        expected_channel = {
            (0.1, 0.1): 0, (0.1, 4.9): 1,
            (4.9, 0.1): 2, (4.9, 4.9): 3
        }[input_state]
        
        # Calculate how close we are to the desired ordering
        target_output = outputs[expected_channel]
        other_outputs = [v for i, v in enumerate(outputs) if i != expected_channel]
        
        # Give partial credit based on relative position
        rank = sum(1 for x in outputs if x > target_output)
        rank_score = max(0, (3 - rank) * 2)  # 6 points for being highest, 4 for second, 2 for third
        
        # Add separation bonus
        separation = target_output - max(other_outputs)
        separation_score = max(0, min(separation * 5, 4))  # Up to 4 additional points for separation
        
        return rank_score + separation_score

    def evaluate_configuration(self, config: Dict[str, float]) -> float:
        """Evaluate full configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return sum(self.evaluate_single_input(config_str, input_state) 
                  for input_state in INPUT_COMBINATIONS)
    
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
        """Intelligent crossover that preserves channel-specific patterns"""
        # Group heaters by their influence on each channel
        channel_groups = {
            'ch1': range(0, 8),    # Heaters likely affecting channel 1
            'ch2': range(8, 16),   # Heaters likely affecting channel 2
            'ch3': range(16, 24),  # Heaters likely affecting channel 3
            'ch4': range(24, 33)   # Heaters likely affecting channel 4
        }
        
        child1, child2 = {}, {}
        
        # For each channel group, randomly choose whether to take the pattern from parent1 or parent2
        for group_heaters in channel_groups.values():
            if random.random() < 0.5:
                # Take this channel's pattern from parent1
                for h in group_heaters:
                    child1[str(h)] = parent1[str(h)]
                    child2[str(h)] = parent2[str(h)]
            else:
                # Take this channel's pattern from parent2
                for h in group_heaters:
                    child1[str(h)] = parent2[str(h)]
                    child2[str(h)] = parent1[str(h)]
        
        # Handle fixed layer heaters
        for h in range(33, 40):
            if random.random() < 0.5:
                child1[str(h)] = parent1[str(h)]
                child2[str(h)] = parent2[str(h)]
            else:
                child1[str(h)] = parent2[str(h)]
                child2[str(h)] = parent1[str(h)]
        
        return child1, child2
    
    def adaptive_mutation(self, config: Dict[str, float], rate: float, generation: int) -> Dict[str, float]:
        result = config.copy()
        temperature = max(1.0 - (generation / 50), 0.1)
        
        for heater in self.config_manager.modifiable_heaters:
            if random.random() < rate * temperature:
                current_value = result[str(heater)]
                
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



    def optimize(self, 
                pop_size: int = 40,
                generations: int = 50,
                base_mutation_rate: float = 0.1,
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
                0.4
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
            if no_improvement_count >= early_stop // 2:
                print("Implementing partial population restart...")
                population[pop_size//2:] = [
                    self.config_manager.generate_random_config() 
                    for _ in range(pop_size//2)
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
            pop_size=20,
            generations=25,
            base_mutation_rate=0.1,
            early_stop=25
        )
        
        print("\nOptimization Complete!")
        print(f"Best Score: {best_score}")
        print(f"Best Configuration: {best_config}")
        
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        hardware.cleanup()

if __name__ == "__main__":
    main()