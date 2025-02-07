import serial
import time
import pyvisa
import random
import numpy as np

class SimpleDecoderOptimizer:
    def __init__(self, population_size=30, mutation_rate=0.2):
        # Configuration
        self.SERIAL_PORT = 'COM4'
        self.BAUD_RATE = 115200
        self.INPUT_HEATERS = [35, 36, 37]
        self.INPUT_STATES = [
            (0.1, 0.1, 0.1),  # 000
            (0.1, 0.1, 4.9),  # 001
            (0.1, 4.9, 0.1),  # 010
            (0.1, 4.9, 4.9),  # 011
            (4.9, 0.1, 0.1),  # 100
            (4.9, 0.1, 4.9),  # 101
            (4.9, 4.9, 0.1)   # 110
        ]
        
        # GA parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Initialize hardware
        self._init_hardware()
        
        # Initialize heater ranges
        self.modifiable_heaters = [i for i in range(40) if i not in self.INPUT_HEATERS]
        self.fixed_first_layer = list(range(33, 40))
        
    def _init_hardware(self):
        """Initialize scopes and serial connection"""
        # Initialize scopes
        rm = pyvisa.ResourceManager()
        self.scope1 = rm.open_resource('USB0::0x1AB1::0x0610::HDO1B244000779::INSTR')
        self.scope2 = rm.open_resource('USB0::0x1AB1::0x0610::HDO1B244100809::INSTR')
        
        for scope in [self.scope1, self.scope2]:
            scope.timeout = 5000
            for i in range(1, 5):
                scope.write(f':CHANnel{i}:DISPlay ON')
                scope.write(f':CHANnel{i}:SCALe 2')
                scope.write(f':CHANnel{i}:OFFSet -6')
        
        # Initialize serial
        self.serial = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE)
        time.sleep(1)
    
    def measure_outputs(self):
        """Measure outputs from both oscilloscopes"""
        outputs = []
        
        # First scope - 4 channels
        for channel in range(1, 5):
            value = float(self.scope1.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
            outputs.append(round(value, 5))
        
        # Second scope - 3 channels
        for channel in range(1, 4):
            value = float(self.scope2.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
            outputs.append(round(value, 5))
            
        return outputs
    
    def send_heater_values(self, config):
        """Send heater values via serial"""
        message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
        self.serial.write(message.encode())
        self.serial.flush()
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
        
        for input_state in self.INPUT_STATES:
            # Apply input state
            config = individual.copy()
            for i, value in enumerate(input_state):
                config[self.INPUT_HEATERS[i]] = value
            
            self.send_heater_values(config)
            time.sleep(0.25)
            outputs = self.measure_outputs()
            
            if None in outputs:
                return -1000
            
            # Get expected and actual highest outputs
            expected_highest = expected_outputs[input_state]
            actual_highest = outputs.index(max(outputs))
            
            # Score based on correct output
            if actual_highest == expected_highest:
                # Base score for correct output
                state_score = 50
                
                # Bonus for separation from other outputs
                max_output = outputs[actual_highest]
                other_outputs = outputs.copy()
                other_outputs.pop(actual_highest)
                separation = max_output - max(other_outputs)
                
                # Add separation bonus
                state_score += min(50, separation * 20)
            else:
                # Penalty for incorrect output
                state_score = -25
            
            total_score += state_score
            
        return total_score
    
    def create_individual(self):
        """Create a random individual"""
        individual = {}
        
        for heater in self.modifiable_heaters:
            if heater in self.fixed_first_layer:
                individual[heater] = 0.01
            else:
                individual[heater] = round(random.uniform(0.1, 4.9), 2)
        
        return individual
    
    def crossover(self, parent1, parent2):
        """Simple uniform crossover"""
        child = {}
        
        for heater in self.modifiable_heaters:
            if heater in self.fixed_first_layer:
                child[heater] = 0.01
            else:
                # 50% chance from each parent
                child[heater] = parent1[heater] if random.random() < 0.5 else parent2[heater]
        
        return child
    
    def mutate(self, individual):
        """Simple mutation"""
        mutated = individual.copy()
        
        for heater in self.modifiable_heaters:
            if heater not in self.fixed_first_layer and random.random() < self.mutation_rate:
                mutated[heater] = round(random.uniform(0.1, 4.9), 2)
        
        return mutated
    
    def select_parent(self, population, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        tournament_idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_idx]
        winner_idx = tournament_idx[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]
    
    def optimize(self, generations=50):
        """Main optimization loop"""
        # Create initial population
        population = [self.create_individual() for _ in range(self.population_size)]
        best_individual = None
        best_fitness = float('-inf')
        
        print("\nStarting optimization...")
        print(f"{'Gen':>4} {'Best':>10} {'Avg':>10} {'Min':>10}")
        print("-" * 40)
        
        for generation in range(generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_individual(individual)
                fitness_scores.append(fitness)
            
            # Update best solution
            current_best = max(fitness_scores)
            if current_best > best_fitness:
                best_idx = fitness_scores.index(current_best)
                best_individual = population[best_idx].copy()
                best_fitness = current_best
            
            # Print statistics
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            min_fitness = min(fitness_scores)
            print(f"{generation:4d} {best_fitness:10.2f} {avg_fitness:10.2f} {min_fitness:10.2f}")
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            new_population.append(population[best_idx])
            
            # Create rest of new population
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.select_parent(population, fitness_scores)
                parent2 = self.select_parent(population, fitness_scores)
                
                # Create child
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return best_individual, best_fitness
    
    def cleanup(self):
        """Clean up hardware connections"""
        self.serial.close()
        self.scope1.close()
        self.scope2.close()

def main():
    # Create optimizer
    optimizer = SimpleDecoderOptimizer(
        population_size=30,
        mutation_rate=0.2
    )
    
    try:
        # Run optimization
        print("Starting 3-bit decoder optimization...")
        best_config, best_fitness = optimizer.optimize(generations=50)
        
        print(f"\nOptimization complete!")
        print(f"Best fitness: {best_fitness:.2f}")
        
        # Print and test final configuration
        print("\nBest Heater Configuration:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f}V")
        
        # Test final configuration
        print("\nTesting final configuration:")
        for input_state in optimizer.INPUT_STATES:
            config = best_config.copy()
            for i, value in enumerate(input_state):
                config[optimizer.INPUT_HEATERS[i]] = value
            
            optimizer.send_heater_values(config)
            time.sleep(0.25)
            outputs = optimizer.measure_outputs()
            
            print(f"\nInputs (A,B,C): {input_state}")
            print(f"Outputs: {outputs}")
            print(f"Highest output: Channel {outputs.index(max(outputs)) + 1}")
    
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    main()