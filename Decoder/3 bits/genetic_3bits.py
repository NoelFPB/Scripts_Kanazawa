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
POPULATION_SIZE = 30
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

    def evaluate_individual(self, individual):
        """Evaluate a single individual's fitness with improved gradient scoring"""
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
            
            # Store all voltages for later analysis
            all_voltages.extend(outputs)
            
            # Calculate separations from all other outputs
            other_outputs = outputs.copy()
            other_outputs.pop(expected_highest)
            
            # Calculate separation metrics
            max_other = max(other_outputs)
            separation = expected_voltage - max_other
            relative_separation = expected_voltage / (max_other + 1e-6)
            all_separations.append(separation)
            
            # Base score for the correct output being highest
            if outputs.index(max(outputs)) == expected_highest:
                state_score = 50  # Base score for correct output
                
                # Gradient-friendly separation scoring
                separation_score = min(100, separation * 40)  # Linear scaling up to 2.5V separation
                
                # Gradient-friendly voltage quality scoring
                voltage_score = min(50, expected_voltage * 15)  # Linear scaling up to ~3.3V
                
                state_score += separation_score + voltage_score
                
                # Small bonus for exceeding thresholds
                if separation > 2.5:
                    state_score += 25
                if expected_voltage > 3.0:
                    state_score += 25
            else:
                # Gradient-friendly penalty that considers how close we were
                ranking = sorted(range(len(outputs)), key=lambda k: outputs[k], reverse=True)
                actual_rank = ranking.index(expected_highest)
                
                # Penalty scales with how far off we are in ranking
                rank_penalty = -30 * (actual_rank + 1)
                
                # Add gradient based on voltage difference to encourage improvement
                voltage_diff_score = (expected_voltage - max_other) * 20
                
                state_score = rank_penalty + voltage_diff_score
            
            total_score += state_score
        
        # Global scoring components
        avg_separation = sum(all_separations) / len(all_separations)
        min_separation = min(all_separations)
        
        # Gradient-friendly global separation quality
        separation_factor = min(150, avg_separation * 50)  # Linear scaling of average separation
        min_separation_bonus = min(100, min_separation * 40)  # Reward for worst-case separation
        
        total_score += separation_factor + min_separation_bonus
        
        # Gradient-friendly voltage level scoring
        avg_voltage = sum(all_voltages) / len(all_voltages)
        if avg_voltage < 0.5:
            # Smooth penalty for low voltages
            voltage_penalty = (avg_voltage / 0.5) ** 2
            total_score *= voltage_penalty
        elif avg_voltage > 3.0:
            # Small bonus for strong overall voltage levels
            total_score += 50
        
        # Consistency bonus
        voltage_std = (sum((v - avg_voltage) ** 2 for v in all_voltages) / len(all_voltages)) ** 0.5
        consistency_score = min(100, max(0, (1.0 - voltage_std) * 100))
        total_score += consistency_score
        
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
        # Ensure tournament size does not exceed population size
        tournament_size = min(TOURNAMENT_SIZE, len(population))  
        
        tournament = random.sample(list(enumerate(fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        
        # Ensure valid index
        if winner[0] >= len(population):
            print(f"Warning: Tournament selection index {winner[0]} out of range. Selecting random individual.")
            return random.choice(population)

        return population[winner[0]]


    def crossover(self, parent1, parent2):
        """Perform crossover with varied mixing"""
        if random.random() > CROSSOVER_RATE:
            return parent1.copy()
        
        child = {}
        for heater in self.modifiable_heaters:
            if heater in self.fixed_first_layer:
                child[heater] = 0.01  # Maintain fixed layer
            else:
                if random.random() < 0.3:  # 30% chance to inherit directly
                    child[heater] = parent1[heater] if random.random() < 0.5 else parent2[heater]
                else:  # 70% chance to mix values
                    weight = random.uniform(0.2, 0.8)  # More variability
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
                
                if mutation_type < 0.3:  # Small Gaussian mutation
                    delta = random.gauss(0, 0.2)  # Small adjustments
                    new_value = current_value + delta
                elif mutation_type < 0.6:  # Medium range mutation
                    delta = random.uniform(-0.5, 0.5)
                    new_value = current_value + delta
                else:  # Large reset mutation
                    new_value = random.uniform(0.1, 4.9)
                
                mutated[heater] = round(min(max(new_value, 0.1), 4.9), 2)
        
        return mutated


    def optimize(self, generations=50):
        """Run genetic algorithm optimization with aggressive exploration"""
        population = self.create_initial_population()
        best_individual = None
        best_fitness = -float('inf')
        last_improvement_fitness = -float('inf')
        generations_without_improvement = 0
        improvement_threshold = 1.0
        
        # Track multiple populations
        hall_of_fame = []
        alternative_populations = []  # Store different promising populations
        
        print("\nStarting genetic algorithm optimization...")
        print("=" * 100)
        print(f"{'Gen':>4} {'Best':>10} {'Avg':>10} {'Min':>10} {'Pop Div':>10} {'No Imp':>8} {'Time (s)':>10} {'Total (m)':>10}")
        print("-" * 100)
        
        total_start_time = time.time()
        restart_count = 0
        
        for generation in range(generations):
            generation_start_time = time.time()
            
            # Evaluate current population
            fitness_scores = []
            for idx, individual in enumerate(population):
                fitness = self.evaluate_individual(individual)
                fitness_scores.append(fitness)
            
            # Calculate statistics
            current_best = max(fitness_scores)
            current_avg = sum(fitness_scores) / len(fitness_scores)
            current_min = min(fitness_scores)
            
            # Update hall of fame with diversity check
            best_idx = fitness_scores.index(current_best)
            current_best_individual = population[best_idx].copy()
            
            # Check if this solution is significantly different from existing ones
            is_unique = True
            for hof_ind, _ in hall_of_fame:
                similarity = sum(abs(hof_ind[h] - current_best_individual[h]) 
                            for h in self.modifiable_heaters) / len(self.modifiable_heaters)
                if similarity < 0.5:  # If too similar to existing solution
                    is_unique = False
                    break
            
            if is_unique and (not hall_of_fame or current_best > hall_of_fame[0][1] - 50):
                hall_of_fame.append((current_best_individual, current_best))
                hall_of_fame.sort(key=lambda x: x[1], reverse=True)
                hall_of_fame = hall_of_fame[:15]  # Keep more diverse solutions
            
            # Calculate population diversity
            diversity = 0
            for heater in self.modifiable_heaters:
                if heater not in self.fixed_first_layer:
                    values = [ind[heater] for ind in population]
                    mean_value = sum(values) / len(values)
                    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
                    diversity += (variance ** 0.5)
            diversity /= len(self.modifiable_heaters)
            
            # Update best individual and check for improvement
            if current_best > best_fitness:
                best_individual = current_best_individual
                if current_best > last_improvement_fitness + improvement_threshold:
                    last_improvement_fitness = current_best
                    generations_without_improvement = 0
                    best_fitness = current_best
                    
                    # Store current population if it's promising
                    if len(alternative_populations) < 5:
                        alternative_populations.append(population[:])
                    else:
                        alternative_populations[restart_count % 5] = population[:]
                else:
                    generations_without_improvement += 1
            else:
                generations_without_improvement += 1
            
            # Print status
            generation_time = time.time() - generation_start_time
            total_time = (time.time() - total_start_time) / 60.0
            
            print(f"{generation:4d} {best_fitness:10.2f} {current_avg:10.2f} "
                f"{current_min:10.2f} {diversity:10.2f} {generations_without_improvement:8d} "
                f"{generation_time:10.2f} {total_time:10.2f}")
            
            # Initialize new population
            new_population = []
            
            # Multi-level restart strategy
            if generations_without_improvement >= 10:
                restart_count += 1
                print(f"\nExecuting restart strategy {restart_count % 3 + 1}...")
                
                if restart_count % 3 == 0:  # Strategy 1: Aggressive mutation
                    new_population = []
                    # Keep top solutions with aggressive mutation
                    sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), 
                                                    key=lambda pair: pair[0], 
                                                    reverse=True)]
                    for ind in sorted_pop[:5]:
                        mutated = ind.copy()
                        for heater in self.modifiable_heaters:
                            if heater not in self.fixed_first_layer and random.random() < 0.7:
                                mutated[heater] = round(random.uniform(0.1, 4.9), 2)
                        new_population.append(mutated)
                    
                elif restart_count % 3 == 1:  # Strategy 2: Population hybridization
                    if alternative_populations:
                        # Mix current population with a stored alternative population
                        alt_pop = random.choice(alternative_populations)
                        new_population = []
                        for i in range(5):
                            if i < len(alt_pop):
                                new_population.append(alt_pop[i].copy())
                    
                else:  # Strategy 3: Complete restart with history influence
                    new_population = []
                    if hall_of_fame:
                        # Use hall of fame solutions as seeds
                        for i in range(min(5, len(hall_of_fame))):
                            base = hall_of_fame[i][0].copy()
                            # Apply medium mutation
                            for heater in self.modifiable_heaters:
                                if heater not in self.fixed_first_layer and random.random() < 0.4:
                                    base[heater] = round(random.uniform(0.1, 4.9), 2)
                            new_population.append(base)
                
                # Fill remaining population with new random individuals
                while len(new_population) < POPULATION_SIZE:
                    new_ind = self.create_individual()
                    new_population.append(new_ind)
                
                # Reset improvement counter and adjust parameters
                generations_without_improvement = 0
                self.mutation_rate = 0.4
                
            else:
                # Regular evolution with improved diversity maintenance
                sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), 
                                                key=lambda pair: pair[0], 
                                                reverse=True)]
                
                # Adaptive elitism
                elite_size = max(2, min(5, int(POPULATION_SIZE * (1 - diversity))))
                new_population.extend(sorted_pop[:elite_size])
                
                # Generate rest of new population
                while len(new_population) < POPULATION_SIZE:
                    if random.random() < 0.9:  # Standard breeding
                        parent1 = self.tournament_selection(population, fitness_scores)
                        parent2 = self.tournament_selection(population, fitness_scores)
                        child = self.crossover(parent1, parent2)
                        child = self.mutate(child)
                    else:  # Occasional fresh blood
                        child = self.create_individual()
                    new_population.append(child)
            
            # Update population
            population = new_population
            
            # Adaptive parameter adjustment
            if diversity < 0.3:
                self.mutation_rate = min(0.5, self.mutation_rate * 1.2)
            elif diversity > 0.6:
                self.mutation_rate = max(0.1, self.mutation_rate * 0.8)
        
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
        best_config, best_fitness = optimizer.optimize(generations=200)
        
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