import numpy as np
import serial
import time
import pyvisa
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

# Configuration Constants
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

# GA Parameters with wider ranges for adaptation
GA_CONFIG = {
    'population_size': 40,
    'num_populations': 3,
    'elite_size': 4,
    'tournament_size': 3,
    'mutation_rate_range': (0.05, 0.4),
    'crossover_rate_range': (0.6, 0.9),
    'initialization_diversity_threshold': 0.3,
    'niche_radius': 0.5,
    'fitness_sharing_alpha': 1.0,
    'stagnation_threshold': 15,
    'migration_interval': 5,
    'migration_size': 2
}

@dataclass
class Population:
    """Class to hold population data and metrics"""
    individuals: List[Dict[int, float]]
    fitness_scores: List[float]
    best_fitness: float = float('-inf')
    best_individual: Optional[Dict[int, float]] = None
    diversity_metric: float = 0.0
    generation: int = 0
    stagnation_counter: int = 0

class DecoderOptimizer:
    def __init__(self):
        """Initialize the optimizer with enhanced tracking and adaptive features"""
        # Hardware initialization
        self.scopes = self._init_scopes()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Initialize heater configuration
        self.modifiable_heaters = [i for i in range(40) if i not in INPUT_HEATERS]
        self.fixed_first_layer = list(range(33, 40))
        
        # GA state tracking
        self.populations = []
        self.global_best = {'fitness': float('-inf'), 'individual': None}
        self.improvement_history = deque(maxlen=50)
        
        # Adaptive parameters
        self.current_mutation_rate = np.mean(GA_CONFIG['mutation_rate_range'])
        self.current_crossover_rate = np.mean(GA_CONFIG['crossover_rate_range'])
        
        # Results tracking
        self.generation_history = []
        self.setup_logging()

    def setup_logging(self):
        """Setup logging for tracking optimization progress"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"optimization_logs_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize logging files
        self.fitness_log = open(f"{self.log_dir}/fitness_history.csv", 'w')
        self.fitness_log.write("Generation,Population,Best,Average,Diversity\n")
        
        self.config_log = open(f"{self.log_dir}/best_configs.json", 'w')

    def _init_scopes(self):
        """Initialize oscilloscopes with error handling"""
        try:
            rm = pyvisa.ResourceManager()
            SCOPE1_ID = 'USB0::0x1AB1::0x0610::HDO1B244000779::INSTR'
            SCOPE2_ID = 'USB0::0x1AB1::0x0610::HDO1B244100809::INSTR'
            
            scopes = []
            for scope_id in [SCOPE1_ID, SCOPE2_ID]:
                scope = rm.open_resource(scope_id)
                scope.timeout = 5000
                for i in range(1, 5):
                    scope.write(f':CHANnel{i}:DISPlay ON')
                    scope.write(f':CHANnel{i}:SCALe 2')
                    scope.write(f':CHANnel{i}:OFFSet -6')
                scopes.append(scope)
            
            return scopes
        except Exception as e:
            print(f"Error initializing scopes: {e}")
            raise

    def create_individual(self, seed: Optional[Dict[int, float]] = None) -> Dict[int, float]:
        """Create a new individual with optional seeding"""
        individual = {}
        
        for heater in self.modifiable_heaters:
            if heater in self.fixed_first_layer:
                individual[heater] = 0.01
            elif seed and random.random() < 0.3:  # 30% chance to inherit from seed
                individual[heater] = seed[heater]
            else:
                # Use different strategies for initialization
                if random.random() < 0.7:  # Standard uniform
                    individual[heater] = round(random.uniform(0.1, 4.9), 2)
                else:  # Biased towards extremes
                    if random.random() < 0.5:
                        individual[heater] = round(random.uniform(0.1, 1.5), 2)
                    else:
                        individual[heater] = round(random.uniform(3.5, 4.9), 2)
        
        return individual

    def initialize_population(self, size: int, seed_individual: Optional[Dict[int, float]] = None) -> List[Dict[int, float]]:
        """Initialize population with diversity enforcement"""
        population = []
        attempts = 0
        max_attempts = size * 10
        
        while len(population) < size and attempts < max_attempts:
            candidate = self.create_individual(seed_individual if random.random() < 0.2 else None)
            
            # Check diversity against existing individuals
            if self.is_sufficiently_diverse(candidate, population):
                population.append(candidate)
            
            attempts += 1
        
        # Fill remaining spots if needed
        while len(population) < size:
            population.append(self.create_individual(seed_individual))
        
        return population

    def is_sufficiently_diverse(self, candidate: Dict[int, float], population: List[Dict[int, float]]) -> bool:
        """Check if candidate is sufficiently different from existing population"""
        if not population:
            return True
            
        for existing in population:
            distance = self.calculate_distance(candidate, existing)
            if distance < GA_CONFIG['initialization_diversity_threshold']:
                return False
        return True

    def calculate_distance(self, ind1: Dict[int, float], ind2: Dict[int, float]) -> float:
        """Calculate normalized distance between two individuals"""
        distances = []
        for heater in self.modifiable_heaters:
            if heater not in self.fixed_first_layer:
                diff = abs(ind1[heater] - ind2[heater])
                normalized_diff = diff / 4.8  # Normalize by voltage range
                distances.append(normalized_diff)
        return sum(distances) / len(distances)

    def calculate_shared_fitness(self, population: List[Dict[int, float]], raw_fitness: List[float]) -> List[float]:
        """Calculate shared fitness scores using modified sharing function"""
        n = len(population)
        shared_fitness = np.zeros(n)
        
        for i in range(n):
            niche_count = 0
            for j in range(n):
                distance = self.calculate_distance(population[i], population[j])
                
                # Modified sharing function with adaptive radius
                if distance < GA_CONFIG['niche_radius']:
                    sh = (1 - (distance / GA_CONFIG['niche_radius']) ** 
                          GA_CONFIG['fitness_sharing_alpha'])
                    niche_count += sh
            
            shared_fitness[i] = raw_fitness[i] / max(1, niche_count)
        
        return shared_fitness

    def tournament_selection(self, population: List[Dict[int, float]], 
                           fitness_scores: List[float], 
                           tournament_size: int) -> Dict[int, float]:
        """Tournament selection with dynamic size adjustment"""
        tournament_size = min(tournament_size, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        
        # Select winner based on tournament
        winner_idx = max(tournament_indices, key=lambda idx: fitness_scores[idx])
        return population[winner_idx].copy()

    def crossover(self, parent1: Dict[int, float], parent2: Dict[int, float]) -> Dict[int, float]:
        """Advanced crossover with multiple strategies"""
        if random.random() > self.current_crossover_rate:
            return parent1.copy()
        
        child = {}
        crossover_type = random.random()
        
        if crossover_type < 0.4:  # Uniform crossover
            for heater in self.modifiable_heaters:
                if heater in self.fixed_first_layer:
                    child[heater] = 0.01
                else:
                    child[heater] = parent1[heater] if random.random() < 0.5 else parent2[heater]
        
        elif crossover_type < 0.7:  # Arithmetic crossover
            weight = random.uniform(0.2, 0.8)
            for heater in self.modifiable_heaters:
                if heater in self.fixed_first_layer:
                    child[heater] = 0.01
                else:
                    value = weight * parent1[heater] + (1 - weight) * parent2[heater]
                    child[heater] = round(min(max(value, 0.1), 4.9), 2)
        
        else:  # Multi-point crossover
            crossover_points = sorted(random.sample(range(len(self.modifiable_heaters)), 2))
            current_parent = parent1
            
            for i, heater in enumerate(self.modifiable_heaters):
                if i in crossover_points:
                    current_parent = parent2 if current_parent == parent1 else parent1
                
                if heater in self.fixed_first_layer:
                    child[heater] = 0.01
                else:
                    child[heater] = current_parent[heater]
        
        return child

    def mutate(self, individual: Dict[int, float]) -> Dict[int, float]:
        """Enhanced mutation with multiple strategies"""
        mutated = individual.copy()
        
        for heater in self.modifiable_heaters:
            if heater not in self.fixed_first_layer and random.random() < self.current_mutation_rate:
                mutation_type = random.random()
                current_value = mutated[heater]
                
                if mutation_type < 0.4:  # Small Gaussian mutation
                    sigma = 0.2 * (1 + self.current_mutation_rate)  # Adaptive magnitude
                    delta = random.gauss(0, sigma)
                    new_value = current_value + delta
                
                elif mutation_type < 0.7:  # Random reset
                    new_value = random.uniform(0.1, 4.9)
                
                else:  # Boundary mutation
                    if current_value < 2.5:
                        new_value = random.uniform(3.5, 4.9)
                    else:
                        new_value = random.uniform(0.1, 1.5)
                
                mutated[heater] = round(min(max(new_value, 0.1), 4.9), 2)
        
        return mutated

    def evaluate_individual(self, individual: Dict[int, float]) -> float:
        """Enhanced evaluation function with multiple scoring components"""
        expected_outputs = {
            (0.1, 0.1, 0.1): 0,  # 000
            (0.1, 0.1, 4.9): 1,  # 001
            (0.1, 4.9, 0.1): 2,  # 010
            (0.1, 4.9, 4.9): 3,  # 011
            (4.9, 0.1, 0.1): 4,  # 100
            (4.9, 0.1, 4.9): 5,  # 101
            (4.9, 4.9, 0.1): 6   # 110
        }
        
        total_score = 0
        all_separations = []
        all_voltages = []
        min_separation = float('inf')
        
        for input_state in INPUT_STATES:
            # Apply input state
            config = individual.copy()
            for i, value in enumerate(input_state):
                config[INPUT_HEATERS[i]] = value
            
            self.send_heater_values(config)
            time.sleep(0.25)
            outputs = self.measure_outputs()
            
            if None in outputs:
                return -float('inf')
            
            expected_highest = expected_outputs[input_state]
            expected_voltage = outputs[expected_highest]
            all_voltages.extend(outputs)
            
            # Calculate separation metrics
            other_outputs = outputs.copy()
            other_outputs.pop(expected_highest)
            max_other = max(other_outputs)
            separation = expected_voltage - max_other
            all_separations.append(separation)
            min_separation = min(min_separation, separation)
            
            # Scoring components
            if outputs.index(max(outputs)) == expected_highest:
                # Base score for correct output
                state_score = 100
                
                # Separation quality
                separation_score = min(150, separation * 60)
                
                # Voltage quality
                voltage_score = min(100, expected_voltage * 20)
                
                # Bonus for exceeding thresholds
                if separation > 2.0:
                    separation_score *= 1.2
                if expected_voltage > 3.0:
                    voltage_score *= 1.2
                
                state_score += separation_score + voltage_score
            else:
                # Penalty based on ranking and separation
                ranking = sorted(range(len(outputs)), key=lambda k: outputs[k], reverse=True)
                actual_rank = ranking.index(expected_highest)
                
                rank_penalty = -50 * (actual_rank + 1)
                separation_penalty = max(-100, separation * 20)  # Less negative if close
                
                state_score = rank_penalty + separation_penalty
            
            total_score += state_score
        
        # Global quality metrics
        avg_separation = sum(all_separations) / len(all_separations)
        separation_consistency = 1 - (max(all_separations) - min(all_separations)) / max(all_separations)
        voltage_consistency = 1 - np.std(all_voltages) / np.mean(all_voltages)
        
        # Global scoring
        global_score = (
            avg_separation * 100 +  # Reward good average separation
            min_separation * 150 +  # Extra reward for good minimum separation
            separation_consistency * 100 +  # Reward consistent separation
            voltage_consistency * 50  # Small reward for voltage consistency
        )
        
        return total_score + global_score

    def send_heater_values(self, config):
        """Send heater configuration to hardware"""
        try:
            message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
            self.serial.write(message.encode())
            self.serial.flush()
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
        except Exception as e:
            print(f"Error sending heater values: {e}")
            raise

    def measure_scope(self, scope_idx, output_queue):
        """Measure outputs from a single scope with error handling"""
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
        """Measure outputs from both scopes in parallel with improved error handling"""
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

    def migrate_between_populations(self, populations: List[Population]):
        """Migrate individuals between populations"""
        migration_size = GA_CONFIG['migration_size']
        
        for i in range(len(populations)):
            # Select best individuals from current population
            sorted_indices = np.argsort(populations[i].fitness_scores)[-migration_size:]
            migrants = [populations[i].individuals[idx].copy() for idx in sorted_indices]
            
            # Send to next population (circular)
            next_pop_idx = (i + 1) % len(populations)
            
            # Replace worst individuals in target population
            worst_indices = np.argsort(populations[next_pop_idx].fitness_scores)[:migration_size]
            for migrant, replace_idx in zip(migrants, worst_indices):
                populations[next_pop_idx].individuals[replace_idx] = migrant

    def update_adaptive_parameters(self, populations: List[Population]):
        """Update mutation and crossover rates based on population state"""
        # Calculate overall improvement rate
        total_improvements = sum(1 for pop in populations 
                               if pop.stagnation_counter < GA_CONFIG['stagnation_threshold'])
        improvement_rate = total_improvements / len(populations)
        
        # Calculate average diversity
        avg_diversity = np.mean([pop.diversity_metric for pop in populations])
        
        # Adjust mutation rate
        if improvement_rate < 0.3 or avg_diversity < 0.2:
            self.current_mutation_rate = min(
                GA_CONFIG['mutation_rate_range'][1],
                self.current_mutation_rate * 1.2
            )
        else:
            self.current_mutation_rate = max(
                GA_CONFIG['mutation_rate_range'][0],
                self.current_mutation_rate * 0.9
            )
        
        # Adjust crossover rate
        if improvement_rate > 0.5:
            self.current_crossover_rate = min(
                GA_CONFIG['crossover_rate_range'][1],
                self.current_crossover_rate * 1.1
            )
        else:
            self.current_crossover_rate = max(
                GA_CONFIG['crossover_rate_range'][0],
                self.current_crossover_rate * 0.95
            )

    def calculate_population_diversity(self, population: List[Dict[int, float]]) -> float:
        """Calculate population diversity metric"""
        if not population:
            return 0.0
            
        n = len(population)
        total_distance = 0
        comparisons = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_distance += self.calculate_distance(population[i], population[j])
                comparisons += 1
        
        return total_distance / max(1, comparisons)

    def optimize(self, generations: int = 200) -> Tuple[Dict[int, float], float]:
        """Main optimization loop with multiple populations"""
        # Initialize multiple populations
        populations = []
        for _ in range(GA_CONFIG['num_populations']):
            pop = Population(
                individuals=self.initialize_population(GA_CONFIG['population_size']),
                fitness_scores=[],
                best_fitness=float('-inf'),
                best_individual=None,
                diversity_metric=0.0
            )
            populations.append(pop)
        
        generation = 0
        global_best_fitness = float('-inf')
        global_best_individual = None
        
        print("\nStarting optimization with multiple populations...")
        print("=" * 100)
        print(f"{'Gen':>4} {'Pop':>4} {'Best':>10} {'Avg':>10} {'Div':>8} {'MutRate':>8} {'CrossRate':>8}")
        print("-" * 100)
        
        try:
            while generation < generations:
                generation_start = time.time()
                
                # Evaluate all populations
                for pop_idx, pop in enumerate(populations):
                    # Evaluate individuals
                    pop.fitness_scores = [self.evaluate_individual(ind) for ind in pop.individuals]
                    
                    # Calculate shared fitness
                    shared_fitness = self.calculate_shared_fitness(pop.individuals, pop.fitness_scores)
                    
                    # Update population metrics
                    current_best_idx = np.argmax(pop.fitness_scores)
                    current_best_fitness = pop.fitness_scores[current_best_idx]
                    
                    if current_best_fitness > pop.best_fitness:
                        pop.best_fitness = current_best_fitness
                        pop.best_individual = pop.individuals[current_best_idx].copy()
                        pop.stagnation_counter = 0
                    else:
                        pop.stagnation_counter += 1
                    
                    # Update global best
                    if current_best_fitness > global_best_fitness:
                        global_best_fitness = current_best_fitness
                        global_best_individual = pop.individuals[current_best_idx].copy()
                    
                    # Create next generation
                    new_population = []
                    
                    # Elitism
                    elite_indices = np.argsort(shared_fitness)[-GA_CONFIG['elite_size']:]
                    new_population.extend([pop.individuals[i].copy() for i in elite_indices])
                    
                    # Generate rest of population
                    while len(new_population) < GA_CONFIG['population_size']:
                        parent1 = self.tournament_selection(
                            pop.individuals, shared_fitness, GA_CONFIG['tournament_size'])
                        parent2 = self.tournament_selection(
                            pop.individuals, shared_fitness, GA_CONFIG['tournament_size'])
                        
                        child = self.crossover(parent1, parent2)
                        child = self.mutate(child)
                        new_population.append(child)
                    
                    # Update population
                    pop.individuals = new_population
                    pop.diversity_metric = self.calculate_population_diversity(pop.individuals)
                    
                    # Log population status
                    avg_fitness = np.mean(pop.fitness_scores)
                    print(f"{generation:4d} {pop_idx:4d} {current_best_fitness:10.2f} "
                          f"{avg_fitness:10.2f} {pop.diversity_metric:8.2f} "
                          f"{self.current_mutation_rate:8.2f} {self.current_crossover_rate:8.2f}")
                    
                    # Log to file
                    self.fitness_log.write(f"{generation},{pop_idx},{current_best_fitness},"
                                         f"{avg_fitness},{pop.diversity_metric}\n")
                    self.fitness_log.flush()
                
                # Migration between populations
                if generation % GA_CONFIG['migration_interval'] == 0:
                    self.migrate_between_populations(populations)
                
                # Update adaptive parameters
                self.update_adaptive_parameters(populations)
                
                # Save best configuration periodically
                if generation % 10 == 0:
                    config_data = {
                        'generation': generation,
                        'fitness': global_best_fitness,
                        'configuration': global_best_individual
                    }
                    json.dump(config_data, self.config_log)
                    self.config_log.write('\n')
                    self.config_log.flush()
                
                generation += 1
                
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
        finally:
            # Close log files
            self.fitness_log.close()
            self.config_log.close()
        
        print("\nOptimization completed!")
        print(f"Best fitness achieved: {global_best_fitness:.2f}")
        return global_best_individual, global_best_fitness

    def cleanup(self):
        """Clean up hardware connections"""
        try:
            self.serial.close()
            for scope in self.scopes:
                scope.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    """Main function to run the optimization"""
    optimizer = DecoderOptimizer()
    try:
        print("Starting 3-bit decoder optimization...")
        best_config, best_fitness = optimizer.optimize(generations=200)
        
        print(f"\nOptimization complete! Final fitness: {best_fitness:.2f}")
        print("\nBest Heater Configuration:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f}V")
        
        # Final testing of best configuration
        print("\nTesting final configuration:")
        for input_state in INPUT_STATES:
            current_config = best_config.copy()
            for i, value in enumerate(input_state):
                current_config[INPUT_HEATERS[i]] = value
            
            optimizer.send_heater_values(current_config)
            time.sleep(0.25)
            outputs = optimizer.measure_outputs()
            
            max_output = max(outputs)
            max_index = outputs.index(max_output)
            
            print(f"\nInputs: {input_state}")
            print(f"Outputs: {outputs}")
            print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
            
            other_outputs = outputs.copy()
            other_outputs.pop(max_index)
            print(f"Separation from next highest: {(max_output - max(other_outputs)):.4f}V")
    
    except Exception as e:
        print(f"Error during optimization: {e}")
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    main()