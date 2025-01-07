import serial
import time
import pyvisa
import json
import random
from typing import List, Dict, Tuple
import statistics

# Configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600

# Genetic Algorithm Parameters
POPULATION_SIZE = 15
NUM_GENERATIONS = 30     
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8    
ELITE_SIZE = 3

class HeaterGA:
    def __init__(self):
        self.voltage_options = [0.1, 1.0, 2.0, 3.0, 4.0, 4.9]
        self.fixed_first_layer = list(range(33, 40))
        self.input_heaters = [36, 37]
        self.input_combinations = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
        self.modifiable_heaters = [i for i in range(33) if i not in self.input_heaters]
        self.scope = self.init_hardware()
        
    def init_hardware(self):
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000

        # Setup channels
        channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
        for channel in channels:
            scope.write(f':{channel}:DISPlay ON')
            scope.write(f':{channel}:SCALe 2')
            scope.write(f':{channel}:OFFSet -6')
        
        return scope

    def create_individual(self) -> Dict[str, float]:
        """Create a random individual (heater configuration)"""
        config = {str(i): random.choice(self.voltage_options) for i in self.modifiable_heaters}
        for heater in self.fixed_first_layer:
            config[str(heater)] = 0.01
        return config

    def create_initial_population(self) -> List[Dict[str, float]]:
        """Create initial population of heater configurations"""
        return [self.create_individual() for _ in range(POPULATION_SIZE)]

    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Perform crossover between two parents"""
        if random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single point crossover for modifiable heaters
        crossover_point = random.randint(0, len(self.modifiable_heaters))
        for i, heater in enumerate(self.modifiable_heaters):
            if i >= crossover_point:
                child1[str(heater)] = parent2[str(heater)]
                child2[str(heater)] = parent1[str(heater)]
        
        return child1, child2

    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Perform mutation on an individual"""
        mutated = individual.copy()
        for heater in self.modifiable_heaters:
            if random.random() < MUTATION_RATE:
                current_value = mutated[str(heater)]
                available_values = [v for v in self.voltage_options if v != current_value]
                mutated[str(heater)] = random.choice(available_values)
        return mutated

    def send_heater_values(self, ser: serial.Serial, config: Dict[str, float]):
        """Send heater values via serial connection"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        ser.write(voltage_message.encode())
        ser.flush()
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(2)

    def measure_outputs(self) -> List[float]:
        """Measure outputs from oscilloscope"""
        try:
            outputs = []
            for channel in range(1, 5):
                value = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                outputs.append(round(value, 5))
            return outputs
        except Exception as e:
            print(f"Measurement error: {e}")
            return [None] * 4

    def evaluate_fitness(self, ser: serial.Serial, config: Dict[str, float]) -> float:
        """Evaluate fitness of a configuration"""
        total_score = 0
        expected_outputs = {
            (0.1, 0.1): 0,
            (0.1, 4.9): 1,
            (4.9, 0.1): 2,
            (4.9, 4.9): 3
        }

        for input_state in self.input_combinations:
            current_config = config.copy()
            current_config[str(self.input_heaters[0])] = input_state[0]
            current_config[str(self.input_heaters[1])] = input_state[1]
            
            self.send_heater_values(ser, current_config)
           
            outputs = self.measure_outputs()
            if None in outputs:
                return -1000
            
            max_output = max(outputs)
            actual_highest = outputs.index(max_output)
            expected_highest = expected_outputs[input_state]
            
            if actual_highest == expected_highest:
                other_outputs = outputs.copy()
                other_outputs.pop(actual_highest)
                separation = max_output - max(other_outputs)
                total_score += 10 + min(separation * 8, 15)  # Enhanced separation scoring
        
        return total_score

    def select_parents(self, population: List[Dict[str, float]], fitness_scores: List[float]) -> List[Dict[str, float]]:
        """Select parents using tournament selection"""
        selected_parents = []
        
        # First, add elite individuals
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:ELITE_SIZE]
        selected_parents.extend([population[i].copy() for i in elite_indices])
        
        # Then fill the rest using tournament selection
        while len(selected_parents) < POPULATION_SIZE:
            tournament_size = 3
            tournament = random.sample(range(len(population)), tournament_size)
            winner_idx = max(tournament, key=lambda i: fitness_scores[i])
            selected_parents.append(population[winner_idx].copy())
        
        return selected_parents

    def run_genetic_algorithm(self, ser: serial.Serial):
        """Run the genetic algorithm optimization"""
        # Initialize population and storage for best solution
        population = self.create_initial_population()
        best_fitness = float('-inf')
        best_config = None
        generation_stats = []

        try:
            # Main genetic algorithm loop
            for generation in range(NUM_GENERATIONS):
                print(f"\nGeneration {generation + 1}/{NUM_GENERATIONS}")
                
                # Evaluate fitness for all individuals
                fitness_scores = []
                for idx, individual in enumerate(population):
                    print(f"Evaluating individual {idx + 1}/{len(population)}")
                    fitness = self.evaluate_fitness(ser, individual)
                    fitness_scores.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_config = individual.copy()
                        print(f"New best fitness: {best_fitness}")
                        
                        # Save best configuration immediately
                        with open(f"best_config_gen_{generation}.json", 'w') as f:
                            json.dump(best_config, f, indent=4)

                # Record generation statistics
                gen_stats = {
                    'generation': generation,
                    'best_fitness': max(fitness_scores),
                    'avg_fitness': statistics.mean(fitness_scores),
                    'worst_fitness': min(fitness_scores)
                }
                generation_stats.append(gen_stats)
                
                # Save generation statistics
                with open('generation_stats.json', 'w') as f:
                    json.dump(generation_stats, f, indent=4)

                # Selection
                parents = self.select_parents(population, fitness_scores)
                
                # Create new population
                new_population = []
                
                # Add elite individuals
                elite_indices = sorted(range(len(fitness_scores)), 
                                    key=lambda i: fitness_scores[i], 
                                    reverse=True)[:ELITE_SIZE]
                new_population.extend([population[i].copy() for i in elite_indices])
                
                # Create rest of new population
                while len(new_population) < POPULATION_SIZE:
                    parent1, parent2 = random.sample(parents, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    # Mutate children
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    new_population.append(child1)
                    if len(new_population) < POPULATION_SIZE:
                        new_population.append(child2)
                
                population = new_population

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
        except Exception as e:
            print(f"Error during optimization: {e}")
        
        return best_config, best_fitness, generation_stats

def main():
    try:
        # Initialize serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Create and run genetic algorithm
        ga = HeaterGA()
        print("Starting genetic algorithm optimization...")
        
        best_config, best_fitness, stats = ga.run_genetic_algorithm(ser)
        
        print("\nOptimization Complete!")
        print(f"Best fitness: {best_fitness}")
        
        # Save final results
        with open("final_best_configuration.json", 'w') as f:
            json.dump(best_config, f, indent=4)
        
        # Print final configuration
        print("\nFinal Heater Configuration:")
        print("{" + ", ".join(f'"{k}": {best_config[k]:.2f}' for k in sorted(best_config.keys())) + "}")
            
        # Test final configuration
        print("\nTesting final configuration:")
        for input_state in ga.input_combinations:
            current_config = best_config.copy()
            current_config[str(ga.input_heaters[0])] = input_state[0]
            current_config[str(ga.input_heaters[1])] = input_state[1]
            
            ga.send_heater_values(ser, current_config)
            
            outputs = ga.measure_outputs()
            
            print(f"\nInputs (A, B): {input_state}")
            print(f"Outputs: {outputs}")
            if outputs[0] is not None:
                max_output = max(outputs)
                max_index = outputs.index(max_output)
                other_outputs = outputs.copy()
                other_outputs.pop(max_index)
                print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
                print(f"Separation from next highest: {(max_output - max(other_outputs)):.4f}V")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()
        ga.scope.close()
        print("\nProgram terminated.")

if __name__ == "__main__":
    main()