import serial
import time
import pyvisa
import json
import random

# Constants
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600

# Initialize oscilloscope
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

# Heater configuration
heater_values = {i: 0.0 for i in range(40)}
fixed_first_layer = list(range(33, 40))
input_heaters = [36, 37]
input_combinations = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
# Create modifiable_heaters in reverse order
modifiable_heaters = sorted([i for i in range(33) if i not in input_heaters], reverse=True)

# Set fixed first layer
for heater in fixed_first_layer:
    heater_values[heater] = 0.01

# Voltage options - simplified set for faster convergence
voltage_options = [0.1, 2.5, 4.9]

# Initialize population
def initialize_population(pop_size):
    """Generate initial random population."""
    population = []
    for _ in range(pop_size):
        chromo = {str(heater): random.choice(voltage_options) for heater in modifiable_heaters}
        for heater in fixed_first_layer:
            chromo[str(heater)] = 0.01
        population.append(chromo)
    return population

# Fitness evaluation
def evaluate_configuration(ser, config):
    """Evaluate fitness of a configuration."""
    total_score = 0
    
    # Expected highest output for each input combination
    expected_outputs = {
        (0.1, 0.1): 0,  # Output1 should be highest
        (0.1, 4.9): 1,  # Output2 should be highest
        (4.9, 0.1): 2,  # Output3 should be highest
        (4.9, 4.9): 3   # Output4 should be highest
    }
    
    for input_state in input_combinations:
        current_config = config.copy()
        current_config["36"] = input_state[0]
        current_config["37"] = input_state[1]
        
        send_heater_values(ser, current_config)
        outputs = measure_outputs()
        if None in outputs:
            return -1000  # Penalize invalid configuration
        
        # Calculate score
        max_output = max(outputs)
        actual_highest = outputs.index(max_output)
        expected_highest = expected_outputs[input_state]
        if actual_highest == expected_highest:
            separation = max_output - max([v for i, v in enumerate(outputs) if i != actual_highest])
            total_score += 10 + min(separation * 5, 10)
    
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
    
    return best_config, best_score

# Communication functions
def send_heater_values(ser, config):
    """Send heater values via serial."""
    voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(2)

def measure_outputs():
    """Measure outputs from oscilloscope"""
    try:
        outputs = []
        for channel in range(1, 5):
            value = float(scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
            outputs.append(round(value, 5))
        return outputs
    except Exception as e:
        print(f"Measurement error: {e}")
        return [None] * 4
    

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

    finally:
        ser.close()


if __name__ == "__main__":
    main_with_logs()





