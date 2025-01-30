import numpy as np
import serial
import time
import pyvisa
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
# Serial port and core configuration
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

# Genetic Algorithm Parameters
POPULATION_SIZE = 20
ELITE_SIZE = 4
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3

class DecoderOptimizer:
    def __init__(self):
        self.scopes = self._init_scopes()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Initialize heater ranges
        self.modifiable_heaters = [i for i in range(40) if i not in INPUT_HEATERS]
        self.fixed_first_layer = list(range(33, 40))
        
        # Initialize genetic algorithm parameters
        self.mutation_rate = MUTATION_RATE  # Start with default mutation rate
        
    def _init_scopes(self):
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
            
            # First scope has 4 channels, second has 3
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
            #measure_start = time.time()
            output_queue = Queue()
            threads = []
            
            # Start measurement threads for both scopes
            for scope_idx in range(2):
                thread = threading.Thread(target=self.measure_scope, 
                                       args=(scope_idx, output_queue))
                threads.append(thread)
                thread.start()
            
            # Wait for both threads to complete
            for thread in threads:
                thread.join()
            
            # Collect results
            all_outputs = []
            scope_results = {}
            
            # Get results from queue
            while not output_queue.empty():
                scope_idx, outputs = output_queue.get()
                scope_results[scope_idx] = outputs
            
            # Combine results in correct order
            if 0 in scope_results and 1 in scope_results:
                all_outputs.extend(scope_results[0])  # First scope (4 channels)
                all_outputs.extend(scope_results[1])  # Second scope (3 channels)
                #total_time = time.time() - measure_start
                #print(f"Parallel measurement time: {total_time:.3f}s")
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
        time.sleep(0.01)

    def evaluate_individual(self, individual):
        """Evaluate a single individual's fitness"""
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
        min_separation = float('inf')  # Track worst-case separation
        
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
                
            # Find highest output
            max_output = max(outputs)
            actual_highest = outputs.index(max_output)
            expected_highest = expected_outputs[input_state]
            
            # Calculate separation ratio for the expected output
            other_outputs = outputs.copy()
            other_outputs.pop(expected_highest)
            max_other = max(other_outputs)
            separation_ratio = outputs[expected_highest] / (max_other + 1e-6)
            
            # Update minimum separation seen
            min_separation = min(min_separation, separation_ratio)
            
            # Score components
            if actual_highest == expected_highest:
                # Base score for correct output
                correct_bonus = 100
                
                # Separation quality score
                # We want ratio > 1.5 for good separation
                separation_quality = max(0, (separation_ratio - 1) * 50)
                
                # Penalize very low voltages on correct output
                voltage_quality = min(1.0, outputs[expected_highest] / 1.0) * 20
                
                # Add components
                state_score = correct_bonus + separation_quality + voltage_quality
            else:
                # Penalty for incorrect output, but maintain some gradient
                voltage_diff = outputs[expected_highest] - max_output
                state_score = max(-50, voltage_diff * 10)  # Bounded penalty
            
            total_score += state_score
        
        # Global separation quality factor
        separation_factor = max(0, (min_separation - 1) * 100)
        total_score += separation_factor
        
        # Penalize configurations with very low overall voltages
        avg_voltage = sum(outputs) / len(outputs)
        if avg_voltage < 0.5:  # Discourage solutions with very low voltages
            total_score *= (avg_voltage * 2)  # Smooth scaling factor
            
        return total_score

    def create_individual(self):
        """Create a random individual (configuration)"""
        individual = {}
        
        # Set random values for modifiable heaters
        for heater in self.modifiable_heaters:
            if heater in self.fixed_first_layer:
                individual[heater] = 0.01  # Fixed first layer value
            else:
                individual[heater] = round(random.uniform(0.1, 4.9), 2)
                
        return individual

    def create_initial_population(self):
        """Create initial population"""
        return [self.create_individual() for _ in range(POPULATION_SIZE)]

    def tournament_selection(self, population, fitness_scores):
        """Select individual using tournament selection"""
        tournament = random.sample(list(enumerate(fitness_scores)), TOURNAMENT_SIZE)
        winner = max(tournament, key=lambda x: x[1])
        return population[winner[0]]

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        if random.random() > CROSSOVER_RATE:
            return parent1.copy()
            
        child = {}
        for heater in self.modifiable_heaters:
            if heater in self.fixed_first_layer:
                child[heater] = 0.01  # Maintain fixed layer
            else:
                # Random weighted average of parents
                weight = random.random()
                value = weight * parent1[heater] + (1 - weight) * parent2[heater]
                child[heater] = round(min(max(value, 0.1), 4.9), 2)
        
        return child

    def mutate(self, individual):
        """Enhanced mutation with adaptive rates"""
        mutated = individual.copy()
        
        for heater in self.modifiable_heaters:
            if heater not in self.fixed_first_layer and random.random() < self.mutation_rate:
                mutation_type = random.random()
                current_value = mutated[heater]
                
                if mutation_type < 0.4:  # Gaussian mutation
                    delta = random.gauss(0, 0.5)
                    new_value = current_value + delta
                elif mutation_type < 0.7:  # Uniform mutation
                    delta = random.uniform(-0.5, 0.5)
                    new_value = current_value + delta
                else:  # Reset mutation
                    new_value = random.uniform(0.1, 4.9)
                 
                mutated[heater] = round(min(max(new_value, 0.1), 4.9), 2)
        
        return mutated

    def optimize(self, generations=50):
        """Run genetic algorithm optimization"""
        population = self.create_initial_population()
        best_individual = None
        best_fitness = -float('inf')
        last_improvement_fitness = -float('inf')
        generations_without_improvement = 0
        improvement_threshold = 1.0  # Minimum improvement to reset counter
        
        print("\nStarting genetic algorithm optimization...")
        print("=" * 100)
        print(f"{'Gen':>4} {'Best':>10} {'Avg':>10} {'Min':>10} {'Pop Div':>10} {'No Imp':>8} {'Time (s)':>10} {'Total (m)':>10}")
        print("-" * 100)
        
        
        total_start_time = time.time()
        
        for generation in range(generations):
            generation_start_time = time.time()
            
            # Evaluate current population
            fitness_scores = []
            #print("\nStarting individual evaluations:")
            for idx, individual in enumerate(population):
                #ind_start = time.time()
                fitness = self.evaluate_individual(individual)
                #ind_time = time.time() - ind_start
                #print(f"Individual {idx}: {ind_time:.2f}s")
                fitness_scores.append(fitness)
            
            # Calculate statistics
            current_best = max(fitness_scores)
            current_avg = sum(fitness_scores) / len(fitness_scores)
            current_min = min(fitness_scores)
            
            # Calculate population diversity
            diversity = 0
            for heater in self.modifiable_heaters:
                if heater not in self.fixed_first_layer:
                    values = [ind[heater] for ind in population]
                    mean_value = sum(values) / len(values)
                    diff = sum(abs(v - mean_value) for v in values) / len(values)
                    diversity += diff
            diversity /= len(self.modifiable_heaters)
            
            # Update best individual and check for improvement
            if current_best > best_fitness:
                best_individual = population[fitness_scores.index(current_best)].copy()
                
                # Check if improvement is significant
                if current_best > last_improvement_fitness + improvement_threshold:
                    last_improvement_fitness = current_best
                    generations_without_improvement = 0
                    best_fitness = current_best
                else:
                    generations_without_improvement += 1
            else:
                generations_without_improvement += 1
            
            # Calculate timing information
            generation_time = time.time() - generation_start_time
            total_time = (time.time() - total_start_time) / 60.0  # Convert to minutes
            
            # Print status
            print(f"{generation:4d} {best_fitness:10.2f} {current_avg:10.2f} "
                  f"{current_min:10.2f} {diversity:10.2f} {generations_without_improvement:8d} "
                  f"{generation_time:10.2f} {total_time:10.2f}")
            
            # Partial reset when no progress
            if generations_without_improvement >= 15:
                print("\nResetting 50% of the population for exploration!")
                population = sorted_pop[:ELITE_SIZE] + [self.create_individual() for _ in range(POPULATION_SIZE // 2)]

            # Create new population
            new_population = []
            
            # Elitism - carry over best individuals
            sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), 
                                             key=lambda pair: pair[0], 
                                             reverse=True)]
            new_population.extend(sorted_pop[:ELITE_SIZE])
            
            # Fill rest of population with crossover and mutation
            while len(new_population) < POPULATION_SIZE:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
            # Adjust mutation rate based on diversity
            if generations_without_improvement > 10:  # If stuck
                self.mutation_rate = 0.5  # Drastically increase mutations
            elif diversity < 0.3:
                self.mutation_rate = min(0.35, self.mutation_rate * 1.2)  # Increase exploration
            elif diversity > 0.6:
                self.mutation_rate = max(0.05, self.mutation_rate * 0.8)  # Reduce mutations

        
        total_time = (time.time() - total_start_time) / 60.0
        print("\nOptimization completed!")
        print(f"Final best fitness: {best_fitness:.2f}")
        print(f"Total generations: {generation + 1}")
        print(f"Total time: {total_time:.2f} minutes")
        print(f"Average time per generation: {total_time / (generation + 1):.2f} minutes")
        print("=" * 100)
        
        return best_individual, best_fitness

    def cleanup(self):
        """Clean up resources"""
        self.serial.close()
        for scope in self.scopes:
            scope.close()

def main():
    optimizer = DecoderOptimizer()
    try:
        print("Starting 3-bit decoder optimization using genetic algorithm...")
        best_config, best_fitness = optimizer.optimize(generations=500)
        
        print(f"\nOptimization complete! Final fitness: {best_fitness:.2f}")
        print("\nBest Heater Configuration:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f}V")

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
            
            print(f"\nInputs (A, B, C): {input_state}")
            print(f"Outputs: {outputs}")
            print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
            other_outputs = outputs.copy()
            other_outputs.pop(max_index)
            print(f"Separation from next highest: {(max_output - max(other_outputs)):.4f}V")
        
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    main()