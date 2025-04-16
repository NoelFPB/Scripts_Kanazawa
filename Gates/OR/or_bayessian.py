import serial
import time
import pyvisa
import numpy as np
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from itertools import product

# Serial port configuration 
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Heater configuration
INPUT_HEATERS = [36, 37]  # Our OR gate inputs
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in INPUT_HEATERS]

# Input states for OR gate testing
INPUT_COMBINATIONS = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]

# Voltage options for discretization
VOLTAGE_OPTIONS = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.9]

class HybridORGateOptimizer:
    """
    Hybrid optimization combining Bayesian optimization principles with 
    evolutionary computation strategies for efficient discrete optimization
    of a physical OR gate.
    """
    def __init__(self, population_size=10, elite_size=3):
        # Initialize physical hardware connections
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Optimization parameters
        self.population_size = population_size
        self.elite_size = elite_size
        
        # Use a fixed kernel with no optimization to eliminate warnings entirely
        # Fixed parameters based on typical values for this problem
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * \
                 Matern(nu=1.5, length_scale=0.5, length_scale_bounds="fixed") + \
                 WhiteKernel(noise_level=0.1, noise_level_bounds="fixed")
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=0.05,  # Higher alpha for numerical stability
            normalize_y=True, 
            optimizer=None  # Disable kernel optimization completely
        )
        
        # Observation history
        self.configs_evaluated = []
        self.scores_evaluated = []
        
        # Track best solution
        self.best_config = None
        self.best_score = float('-inf')
        
        # Current population
        self.population = []
        self.population_scores = []
        
    def _init_scope(self):
        """Initialize the oscilloscope and set up channels"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000

        # Setup only Channel 1 for OR gate output measurement
        scope.write(':CHANnel1:DISPlay ON')
        scope.write(':CHANnel1:SCALe 2')
        scope.write(':CHANnel1:OFFSet -6')

        # Turn off other channels
        for channel_num in range(2, 5):
            scope.write(f':CHANnel{channel_num}:DISPlay OFF')
            
        return scope
    
    def send_heater_values(self, config):
        """Send heater values to the hardware"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.serial.write(voltage_message.encode())
        self.serial.flush()
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
    
    def measure_output(self):
        """Measure the OR gate output from oscilloscope"""
        try:
            value = float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel1'))
            return round(value, 5)
        except Exception as e:
            print(f"Measurement error: {e}")
            return None
    
    def evaluate_configuration(self, config):
        """Evaluate a configuration across all input combinations with OR gate behavior"""
        # Check if we've already evaluated this configuration
        config_key = tuple(config.get(h, 0.0) for h in sorted(config.keys()))
        for i, past_config in enumerate(self.configs_evaluated):
            past_key = tuple(past_config.get(h, 0.0) for h in sorted(past_config.keys()))
            if config_key == past_key:
                return self.scores_evaluated[i], None
        
        # If not found in history, evaluate on hardware
        total_score = 0
        results = []
        
        # Test each input combination
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            # Send configuration to hardware
            self.send_heater_values(current_config)
            time.sleep(0.25)  # Wait for the system to stabilize
            
            # Measure output
            output = self.measure_output()
            if output is None:
                return -1000, []  # Return a large negative score on error
            
            # OR Gate scoring function - opposite of AND gate for the first case
            if input_state == (0.1, 0.1):
                # Both inputs LOW -> output should be LOW
                if output < 1:  # Stricter threshold for LOW
                    # Exponential reward for lower voltage
                    total_score += 30 + (10 * np.exp(-output))
                elif output < 1.5:
                    # Linear reward for acceptable LOW signal
                    total_score += 15 + (10 * (1.0 - output))
                else:
                    # Severe penalty for incorrect LOW signal
                    total_score -= 30 + (output * output)
            else:
                # Any other input combination -> output should be HIGH
                if output > 3.0:  # Higher threshold for clearer distinction
                    # Quadratic reward for higher voltage - encourages stronger HIGH signal
                    total_score += 30 + (output * output)
                elif output > 2.0:
                    # Linear reward for acceptable HIGH signal
                    total_score += 15 + (output * 5)
                else:
                    # Severe penalty for incorrect HIGH signal
                    total_score -= 30
            
            # Record result
            results.append({
                'inputs': input_state,
                'output': output,
                'expected': 'LOW' if input_state == (0.1, 0.1) else 'HIGH',
                'actual': 'HIGH' if output > 2.0 else 'LOW'
            })
        
        # Add to history
        self.configs_evaluated.append(config.copy())
        self.scores_evaluated.append(total_score)
        
        # Update best if improved
        if total_score > self.best_score:
            self.best_score = total_score
            self.best_config = config.copy()
            print(f"New best score: {self.best_score}")
        
        return total_score, results
    
    def config_to_vector(self, config):
        """Convert a configuration dictionary to a feature vector"""
        return np.array([[config.get(h, 0.0) for h in MODIFIABLE_HEATERS]])
    
    def vector_to_config(self, vector):
        """Convert a feature vector to a configuration dictionary"""
        config = {h: vector[0][i] for i, h in enumerate(MODIFIABLE_HEATERS)}
        
        # Add fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                config[h] = 0.01
                
        return config
    
    def initialize_population(self):
        """Initialize population with diverse configurations"""
        print("Initializing population...")
        self.population = []
        self.population_scores = []
        
        # Use the good AND gate solution as a starting point (with some modifications)
        # This leverages the fact that OR gates often have similar structure to AND gates
        initial_config = {
            0: 0.00, 1: 1.50, 2: 4.00, 3: 4.00, 4: 4.00, 5: 0.00, 6: 4.00, 7: 0.00, 
            8: 0.50, 9: 0.00, 10: 1.50, 11: 3.50, 12: 2.50, 13: 4.90, 14: 3.50, 15: 1.50, 
            16: 1.50, 17: 0.00, 18: 4.50, 19: 4.00, 20: 0.50, 21: 0.00, 22: 0.50, 23: 1.00, 
            24: 4.00, 25: 3.00, 26: 2.50, 27: 1.50, 28: 0.10, 29: 0.50, 30: 3.50, 31: 2.50, 
            32: 2.00, 33: 0.01, 34: 0.01, 35: 0.01, 36: 0.00, 37: 0.00, 38: 0.01, 39: 0.01
        }
        
        score, _ = self.evaluate_configuration(initial_config)
        self.population.append(initial_config)
        self.population_scores.append(score)
        
        # First individual: all zeros (minimal configuration)
        zero_config = {h: 0.0 for h in MODIFIABLE_HEATERS}
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                zero_config[h] = 0.01
        
        score, _ = self.evaluate_configuration(zero_config)
        self.population.append(zero_config)
        self.population_scores.append(score)
        
        # Second individual: all mid-range values
        mid_config = {h: 2.5 for h in MODIFIABLE_HEATERS}
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                mid_config[h] = 0.01
        
        score, _ = self.evaluate_configuration(mid_config)
        self.population.append(mid_config)
        self.population_scores.append(score)
        
        # Random individuals for the rest of the population
        while len(self.population) < self.population_size:
            config = {h: random.choice(VOLTAGE_OPTIONS) for h in MODIFIABLE_HEATERS}
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:
                    config[h] = 0.01
            
            score, _ = self.evaluate_configuration(config)
            self.population.append(config)
            self.population_scores.append(score)
        
    def selection(self):
        """Select parents for reproduction using tournament selection"""
        # Sort population by score (descending)
        sorted_indices = np.argsort(self.population_scores)[::-1]
        
        # Elites automatically survive to next generation
        elites = [self.population[i].copy() for i in sorted_indices[:self.elite_size]]
        
        # Tournament selection for parents
        parents = []
        for _ in range(self.population_size - self.elite_size):
            # Select 3 random individuals for tournament
            candidates = random.sample(range(self.population_size), 3)
            # Choose the one with best score
            tournament_winner = max(candidates, key=lambda idx: self.population_scores[idx])
            parents.append(self.population[tournament_winner].copy())
        
        return elites, parents
    
    def crossover(self, parent1, parent2):
        """Create child through uniform crossover with Bayesian guidance"""
        child = {}
        
        # Add fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                child[h] = 0.01
        
        # For each modifiable heater, choose from either parent
        for h in MODIFIABLE_HEATERS:
            # Bayesian guidance: if we have enough data points
            if len(self.configs_evaluated) >= 10:
                # Create test vectors for both parent values at this heater
                test1 = parent1.copy()
                test2 = parent2.copy()
                
                # Try to predict which one is better
                X = np.vstack([self.config_to_vector(c) for c in self.configs_evaluated])
                y = np.array(self.scores_evaluated)
                
                try:
                    self.gp.fit(X, y)
                    mu1, std1 = self.gp.predict(self.config_to_vector(test1), return_std=True)
                    mu2, std2 = self.gp.predict(self.config_to_vector(test2), return_std=True)
                    
                    # Upper confidence bound (UCB) acquisition
                    ucb1 = mu1 + 1.96 * std1
                    ucb2 = mu2 + 1.96 * std2
                    
                    # Choose the better one based on UCB
                    if ucb1 > ucb2:
                        child[h] = parent1[h]
                    else:
                        child[h] = parent2[h]
                except:
                    # If GP fitting fails, fallback to random selection
                    child[h] = random.choice([parent1[h], parent2[h]])
            else:
                # Not enough data for GP, use random selection
                child[h] = random.choice([parent1[h], parent2[h]])
        
        return child
    
    def mutation(self, config, mutation_rate=0.1, bayesian_guided=True):
        """Mutate configuration with adaptive mutation guided by Bayesian model"""
        mutated = config.copy()
        
        # For each modifiable heater
        for h in MODIFIABLE_HEATERS:
            # Decide whether to mutate this heater
            if random.random() < mutation_rate:
                current_value = mutated[h]
                
                if bayesian_guided and len(self.configs_evaluated) >= 10:
                    try:
                        # Fit GP model
                        X = np.vstack([self.config_to_vector(c) for c in self.configs_evaluated])
                        y = np.array(self.scores_evaluated)
                        self.gp.fit(X, y)
                        
                        # Test each possible voltage value
                        candidates = []
                        ucb_scores = []
                        
                        for voltage in VOLTAGE_OPTIONS:
                            if voltage != current_value:
                                test_config = mutated.copy()
                                test_config[h] = voltage
                                
                                # Predict performance
                                mu, std = self.gp.predict(self.config_to_vector(test_config), return_std=True)
                                ucb = mu + 1.96 * std
                                
                                candidates.append(voltage)
                                ucb_scores.append(ucb[0])
                        
                        # Select voltage with highest UCB
                        if candidates:
                            best_idx = np.argmax(ucb_scores)
                            mutated[h] = candidates[best_idx]
                    except:
                        # Fallback to random mutation if GP fails
                        mutated[h] = random.choice([v for v in VOLTAGE_OPTIONS if v != current_value])
                else:
                    # Regular random mutation
                    mutated[h] = random.choice([v for v in VOLTAGE_OPTIONS if v != current_value])
        
        return mutated
    
    def evolve_population(self):
        """Create next generation through selection, crossover and mutation"""
        # Select elites and parents
        elites, parents = self.selection()
        
        # Create new population starting with elites
        new_population = elites.copy()
        
        # Create children to fill the rest of the population
        while len(new_population) < self.population_size:
            # Select two parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Create child through crossover
            child = self.crossover(parent1, parent2)
            
            # Mutate child
            child = self.mutation(child)
            
            # Add to new population
            new_population.append(child)
        
        # Evaluate new individuals
        new_scores = []
        for config in new_population:
            score, _ = self.evaluate_configuration(config)
            new_scores.append(score)
        
        # Update population
        self.population = new_population
        self.population_scores = new_scores
    
    def hill_climbing_refinement(self, config, iterations=3):
        """Refine solution with hill climbing"""
        print("\nRefining best solution with hill climbing...")
        best_config = config.copy()
        best_score, _ = self.evaluate_configuration(best_config)
        
        for iteration in range(iterations):
            improved = False
            
            # Try to improve each heater one by one
            for h in MODIFIABLE_HEATERS:
                current_value = best_config[h]
                
                # Try all other voltage options
                for new_value in [v for v in VOLTAGE_OPTIONS if v != current_value]:
                    test_config = best_config.copy()
                    test_config[h] = new_value
                    
                    score, _ = self.evaluate_configuration(test_config)
                    
                    if score > best_score:
                        best_score = score
                        best_config = test_config.copy()
                        improved = True
                        print(f"  Refinement improved score to: {best_score}")
                        break
                
                if improved:
                    break
            
            if not improved:
                print("  No further improvements found")
                break
        
        return best_config, best_score
    
    def optimize(self, generations=7):
        """Run hybrid optimization for OR gate configuration"""
        print("Starting hybrid optimization for OR gate...")
        
        # Initialize population
        self.initialize_population()
        
        # Run evolutionary algorithm with Bayesian guidance
        for generation in range(generations):
            print(f"\nGeneration {generation + 1}/{generations}")
            print(f"Current best score: {self.best_score}")
            
            # Evolve population
            self.evolve_population()
            
            # Display statistics
            avg_score = np.mean(self.population_scores)
            max_score = np.max(self.population_scores)
            print(f"Population stats - Avg score: {avg_score:.2f}, Max score: {max_score:.2f}")
        
        # Final refinement with hill climbing
        final_config, final_score = self.hill_climbing_refinement(self.best_config)
        
        return final_config, final_score
    
    def test_final_configuration(self, config):
        """Test and print performance of the optimized configuration"""
        print("\nTesting final OR gate configuration:")
        all_correct = True
        
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            output_value = self.measure_output()
            
            is_high = output_value > 2.0
            expected_high = input_state != (0.1, 0.1)  # Only (Low,Low) should be LOW
            is_correct = is_high == expected_high
            
            if not is_correct:
                all_correct = False
            
            print(f"\nInputs (A, B): {input_state}")
            print(f"OR Output: {output_value:.4f}V")
            print(f"Output is: {'HIGH' if is_high else 'LOW'}")
            print(f"Expected: {'HIGH' if expected_high else 'LOW'}")
            print(f"Correct: {'Yes' if is_correct else 'No'}")
        
        if all_correct:
            print("\nSuccess! The OR gate is working correctly for all input combinations.")
        else:
            print("\nThe OR gate is not working perfectly for all input combinations.")
        
        return all_correct
    
    def cleanup(self):
        """Close connections"""
        self.serial.close()
        self.scope.close()
        print("Connections closed.")


def main():
    optimizer = HybridORGateOptimizer(population_size=8, elite_size=2)
    
    try:
        # Run hybrid optimization
        best_config, best_score = optimizer.optimize(generations=7)
        
        print("\nOptimization complete!")
        print(f"Best score: {best_score}")
        
        # Print final heater configuration
        print("\nFinal Heater Configuration:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f}V")
        
        # Test final configuration
        optimizer.test_final_configuration(best_config)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    main()