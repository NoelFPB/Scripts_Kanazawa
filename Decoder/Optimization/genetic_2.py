import serial
import time
import pyvisa
import json
import random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

# Enhanced Constants
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600
CACHE_SIZE = 1024
VOLTAGE_OPTIONS = [0.1, 1.0, 2.5, 3.7, 4.9]  # Expanded voltage options
INPUT_COMBINATIONS = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]

class ConfigurationManager:
    """Manages configuration generation and validation"""
    def __init__(self):
        self.modifiable_heaters = sorted([i for i in range(33) if i not in [36, 37]], reverse=True)
        self.fixed_first_layer = list(range(33, 40))
    
    def generate_random_config(self) -> Dict[str, float]:
        """Generate a random configuration with improved diversity"""
        config = {
            str(h): random.choice(VOLTAGE_OPTIONS) 
            for h in self.modifiable_heaters
        }
        # More diverse initialization for fixed layer
        config.update({
            str(h): random.choice([0.01, 0.1, 0.5]) 
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
        time.sleep(2)  # Reduced delay
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
        """Evaluate single input combination with caching"""
        config = json.loads(config_str)
        config["36"] = input_state[0]
        config["37"] = input_state[1]
        
        self.hardware.send_heater_values(config)
        outputs = self.hardware.measure_outputs()
        
        if None in outputs:
            return -10.0
        
        expected_highest = {
            (0.1, 0.1): 0, (0.1, 4.9): 1,
            (4.9, 0.1): 2, (4.9, 4.9): 3
        }[input_state]
        
        max_output = max(outputs)
        actual_highest = outputs.index(max_output)
        
        if actual_highest != expected_highest:
            return 0.0
        
        separation = max_output - max(v for i, v in enumerate(outputs) if i != actual_highest)
        return min(separation * 5, 10)
    
<<<<<<< HEAD
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
=======
    return total_score

# Selection
def select_parents(population, fitness, num_parents):
    """Select parents using roulette wheel selection."""
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    parents = random.choices(population, probabilities, k=num_parents)
    return parents

# Crossover
def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    crossover_point = random.randint(1, len(modifiable_heaters) - 1)
    child1 = {**{k: parent1[k] for k in list(parent1)[:crossover_point]},
              **{k: parent2[k] for k in list(parent2)[crossover_point:]}}
    child2 = {**{k: parent2[k] for k in list(parent2)[:crossover_point]},
              **{k: parent1[k] for k in list(parent1)[crossover_point:]}}
    return child1, child2

# Mutation
def mutate(chromo, mutation_rate):
    """Mutate a chromosome."""
    for heater in modifiable_heaters:
        if random.random() < mutation_rate:
            chromo[str(heater)] = random.choice(voltage_options)
    return chromo

def select_parents_tournament(population, fitness, num_parents, tournament_size=3):
    parents = []
    for _ in range(num_parents):
        competitors = random.sample(list(zip(population, fitness)), tournament_size)
        winner = max(competitors, key=lambda x: x[1])
        parents.append(winner[0])
    return parents


# GA main loop
def genetic_algorithm_with_logs(ser, pop_size=10, generations=30, mutation_rate=0.05):
    """Run the Genetic Algorithm with optimizations."""
    population = initialize_population(pop_size)
    fitness_cache = {}
    
    def evaluate_with_cache(config):
        config_tuple = tuple(sorted(config.items()))
        if config_tuple in fitness_cache:
            return fitness_cache[config_tuple]
        fitness = evaluate_configuration(ser, config)
        fitness_cache[config_tuple] = fitness
        return fitness
    
    fitness = [evaluate_with_cache(chromo) for chromo in population]
    
    for generation in range(generations):
        print(f"\n===== Generation {generation + 1}/{generations} =====")

        # Select parents
        parents = select_parents_tournament(population, fitness, pop_size // 2)
        
        # Generate offspring
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = crossover(parent1, parent2)
                offspring.extend([child1, child2])

        # Mutate offspring
        offspring = [mutate(chromo, mutation_rate) for chromo in offspring]

        # Evaluate offspring fitness
        offspring_fitness = [evaluate_with_cache(chromo) for chromo in offspring]

        # Update population
        combined_population = parents + offspring
        combined_fitness = [evaluate_with_cache(chromo) for chromo in combined_population]

        # Select the next generation
        sorted_indices = sorted(range(len(combined_population)), key=lambda i: combined_fitness[i], reverse=True)
        population = [combined_population[i] for i in sorted_indices[:pop_size]]
        fitness = [combined_fitness[i] for i in sorted_indices[:pop_size]]

        # Log best configuration and score
        best_config, best_score = population[0], fitness[0]
        print(f"Best configuration in generation {generation + 1}: {json.dumps(best_config, indent=2)}")
        print(f"Best score: {best_score}")

        # Early stopping if no improvement
        if generation > 5 and fitness[0] == fitness[-1]:
            print("Early stopping due to no improvement.")
            break
>>>>>>> 4efc5e5393615a789086e0ee62e4b82931a736bd
    
    def adaptive_crossover(self, parent1: Dict[str, float], 
                          parent2: Dict[str, float],
                          generation: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Adaptive crossover with varying points"""
        keys = list(parent1.keys())
        num_points = min(3, 1 + generation // 15)  # Increase crossover points over time
        points = sorted(random.sample(range(len(keys)), num_points))
        
        child1, child2 = {}, {}
        swap = True
        current_point = 0
        
        for i, key in enumerate(keys):
            if current_point < len(points) and i >= points[current_point]:
                swap = not swap
                current_point += 1
            
            if swap:
                child1[key], child2[key] = parent2[key], parent1[key]
            else:
                child1[key], child2[key] = parent1[key], parent2[key]
        
        return child1, child2
    
<<<<<<< HEAD
    def adaptive_mutation(self, config: Dict[str, float], 
                         rate: float, 
                         generation: int) -> Dict[str, float]:
        """Adaptive mutation with local search"""
        result = config.copy()
        
        for heater in self.config_manager.modifiable_heaters:
            if random.random() < rate:
                current_value = result[str(heater)]
                # Local search: prefer nearby values
                if random.random() < 0.7:
                    idx = VOLTAGE_OPTIONS.index(current_value)
                    if idx > 0 and idx < len(VOLTAGE_OPTIONS) - 1:
                        result[str(heater)] = random.choice(
                            [VOLTAGE_OPTIONS[idx-1], VOLTAGE_OPTIONS[idx+1]]
                        )
                else:
                    result[str(heater)] = random.choice(VOLTAGE_OPTIONS)
        
        return result
    
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
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        hardware = HardwareInterface()
        optimizer = GeneticOptimizer(hardware, config_manager)
        
        # Run optimization
        best_config, best_score = optimizer.optimize(
            pop_size=40,
            generations=50,
            base_mutation_rate=0.1,
            early_stop=15
        )
        
        print("\nOptimization Complete!")
        print(f"Best Score: {best_score}")
        print(f"Best Configuration: {json.dumps(best_config, indent=2)}")
        
        # Save results with history
        results = {
            "best_config": best_config,
            "best_score": best_score,
            "optimization_history": [
                {"config": config, "score": score}
                for config, score in optimizer.best_configs_history
            ]
        }
        
        with open("optimization_results.json", 'w') as f:
            json.dump(results, f, indent=4)
            
    except Exception as e:
        print(f"Error: {e}")
=======

def main_with_logs():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

        print("Starting Genetic Algorithm with logging...")
        best_config, best_score = genetic_algorithm_with_logs(ser)

        print("\nOptimization Complete!")
        print(f"Best Score: {best_score}")
        print("Best Configuration:")
        print(json.dumps(best_config, indent=2))

        # Save configuration
        with open("best_configuration_with_logs.json", 'w') as f:
            json.dump(best_config, f, indent=4)

    except Exception as e:
        print(f"Error: {e}")

>>>>>>> 4efc5e5393615a789086e0ee62e4b82931a736bd
    finally:
        hardware.cleanup()


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main_with_logs()





>>>>>>> 4efc5e5393615a789086e0ee62e4b82931a736bd
