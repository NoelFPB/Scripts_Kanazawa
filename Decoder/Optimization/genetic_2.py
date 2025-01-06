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

# GA main loop
def genetic_algorithm(ser, pop_size=20, generations=50, mutation_rate=0.1):
    """Run the Genetic Algorithm."""
    # Initialize population
    population = initialize_population(pop_size)
    fitness = [evaluate_configuration(ser, chromo) for chromo in population]
    
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        
        # Select parents
        parents = select_parents(population, fitness, pop_size // 2)
        
        # Generate offspring through crossover
        offspring = []
        for i in range(0, len(parents), 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            offspring.extend([child1, child2])
        
        # Apply mutation
        offspring = [mutate(chromo, mutation_rate) for chromo in offspring]
        
        # Evaluate offspring fitness
        offspring_fitness = [evaluate_configuration(ser, chromo) for chromo in offspring]
        
        # Combine population and offspring
        population = parents + offspring
        fitness = [evaluate_configuration(ser, chromo) for chromo in population]
        
        # Track best configuration
        best_index = fitness.index(max(fitness))
        best_config = population[best_index]
        best_score = fitness[best_index]
        
        print(f"Best score in generation: {best_score}")
    
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
    
# Main function
def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        
        # Run Genetic Algorithm
        best_config, best_score = genetic_algorithm(ser)
        
        print("\nOptimization Complete!")
        print(f"Best Score: {best_score}")
        print(f"Best Configuration: {best_config}")
        
        # Save configuration
        with open("best_configuration.json", 'w') as f:
            json.dump(best_config, f, indent=4)
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        ser.close()

if __name__ == "__main__":
    main()
