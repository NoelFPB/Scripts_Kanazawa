import serial
import time
import pyvisa
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution
import random
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
def create_voltage_options(start=0.1, end=4.9, step=0.5):
    options = []
    current = start
    
    while current <= end:
        options.append(round(current, 1))  # Round to 1 decimal place to avoid floating point issues
        current += step
    
    # Add the final value (4.9) if it's not already included
    if end not in options:
        options.append(end)
        
    return options

# Example usage with default step size of 0.5V
VOLTAGE_OPTIONS = create_voltage_options()
print(VOLTAGE_OPTIONS)




class ORGateOptimizer:
    """
    Optimize a physical OR gate using ensemble learning and evolutionary strategies.
    """
    def __init__(self):
        # Initialize hardware connections
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Initialize Random Forest model for surrogate modeling
        # Random Forest is more robust to noisy data than Gaussian Process
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            min_samples_split=3,
            random_state=42
        )
        
        # History of evaluated configurations
        self.configs_evaluated = []
        self.scores_evaluated = []
        
        # Best configuration found
        self.best_config = None
        self.best_score = float('-inf')
        
    def _init_scope(self):
        """Initialize oscilloscope for OR gate output measurement"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000

        # Setup Channel 1 for OR gate output measurement
        scope.write(':CHANnel1:DISPlay ON')
        scope.write(':CHANnel1:SCALe 2')
        scope.write(':CHANnel1:OFFSet -6')

        # Turn off other channels
        for channel_num in range(2, 5):
            scope.write(f':CHANnel{channel_num}:DISPlay OFF')
            
        return scope
    
    def send_heater_values(self, config):
        """Send heater voltage configuration to hardware"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.serial.write(voltage_message.encode())
        self.serial.flush()
        time.sleep(0.01)  # Small delay to ensure message is sent
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
    
    def measure_output(self):
        """Measure the OR gate output voltage from oscilloscope"""
        try:
            value = float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel1'))
            return round(value, 5)
        except Exception as e:
            print(f"Measurement error: {e}")
            return None
    
    def config_to_vector(self, config):
        """Convert configuration dictionary to feature vector for model"""
        return np.array([config.get(h, 0.1) for h in MODIFIABLE_HEATERS])
    
    def vector_to_config(self, vector):
        """Convert feature vector to configuration dictionary"""
        config = {h: vector[i] for i, h in enumerate(MODIFIABLE_HEATERS)}
        
        # Add fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                config[h] = 0.01
                
        return config
    
    def evaluate_configuration(self, config):
        """
        Evaluate a configuration for OR gate behavior.
        OR gate truth table:
        0 OR 0 = 0 (LOW)
        0 OR 1 = 1 (HIGH)
        1 OR 0 = 1 (HIGH)
        1 OR 1 = 1 (HIGH)
        """
        # Check if we've already evaluated this configuration
        config_key = tuple(config.get(h, 0.1) for h in sorted(config.keys()))
        for i, past_config in enumerate(self.configs_evaluated):
            past_key = tuple(past_config.get(h, 0.1) for h in sorted(past_config.keys()))
            if config_key == past_key:
                return self.scores_evaluated[i], None
        
        # If not in history, evaluate on hardware
        total_score = 0
        results = []
        
        # Test each input combination
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            # Send configuration to hardware
            self.send_heater_values(current_config)
            time.sleep(0.2)  # Wait for the system to stabilize
            
            # Measure output
            output = self.measure_output()
            if output is None:
                return -1000, []  # Return a large negative score on error
            
            # NAND Gate scoring function
            if input_state == (4.9, 4.9):
                # Both inputs HIGH -> output should be LOW
                if output < 1:
                    # Stronger reward for cleaner LOW signal
                    total_score += 40 + (20 * np.exp(-output))
                elif output < 1.5:
                    # Acceptable but not optimal
                    total_score += 20 + (10 * (1.5 - output))
                else:
                    # Severe penalty for incorrect behavior
                    total_score -= 50 + (output * output)
            else:
                # Any other input combination -> output should be HIGH
                if output > 4.2 and output < 6:
                    # Stronger reward for cleaner HIGH signal
                    total_score += 40 + (5 * output)
                elif output > 3.7 and output < 6:
                    # Acceptable but not optimal
                    total_score += 20 + (5 * output)
                else:
                    # Severe penalty for incorrect behavior
                    total_score -= 50 + ((5.0 - output) * (5.0 - output))
            
            # Record results
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
            print(f"New best score: {total_score:.2f}")
        
        return total_score, results
    
    def initial_sampling(self, n_samples=10):
        """Initial sampling phase using different strategies"""
        print(f"Performing initial sampling with {n_samples} configurations...")
        
        # Zero configuration (baseline)
        zero_config = {h: 0.1 for h in MODIFIABLE_HEATERS}
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                zero_config[h] = 0.01
        
        score, _ = self.evaluate_configuration(zero_config)
        print(f"Zero configuration: Score = {score:.2f}")
        
        # Random sampling
        for i in range(n_samples - 1):
            # Generate configuration with structured patterns
            if i % 3 == 0:
                # Alternating pattern
                config = {h: VOLTAGE_OPTIONS[h % len(VOLTAGE_OPTIONS)] for h in MODIFIABLE_HEATERS}
            elif i % 3 == 1:
                # Gradient pattern
                config = {h: VOLTAGE_OPTIONS[min(h // 3, len(VOLTAGE_OPTIONS)-1)] for h in MODIFIABLE_HEATERS}
            else:
                # Random pattern
                config = {h: random.choice(VOLTAGE_OPTIONS) for h in MODIFIABLE_HEATERS}
            
            # Add fixed first layer values
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:
                    config[h] = 0.01
            
            score, _ = self.evaluate_configuration(config)
            print(f"Initial configuration {i+1}/{n_samples-1}: Score = {score:.2f}")
    
    def train_surrogate_model(self):
        """Train surrogate model on collected data"""
        if len(self.configs_evaluated) < 5:
            print("Not enough data to train surrogate model yet.")
            return False
        
        X = np.array([self.config_to_vector(c) for c in self.configs_evaluated])
        y = np.array(self.scores_evaluated)
        
        # Train the random forest model
        self.model.fit(X, y)
        
        # Validate model
        predictions = self.model.predict(X)
        r2 = np.corrcoef(predictions, y)[0, 1] ** 2
        print(f"Surrogate model trained. R² = {r2:.4f}")
        
        return True
    
    def predict_score(self, config):
        """Predict score for a configuration using surrogate model"""
        X = self.config_to_vector(config).reshape(1, -1)
        return self.model.predict(X)[0]
    
    def evolution_step(self, population_size=10, generations=3):
        """Run mini evolutionary algorithm guided by surrogate model"""
        print("\nRunning evolutionary optimization step...")
        
        # Create initial population from best configurations and random ones
        population = []
        
        # Add best config so far
        if self.best_config:
            population.append(self.best_config.copy())
        
        # Add top configurations from history
        if len(self.configs_evaluated) > 1:
            sorted_indices = np.argsort(self.scores_evaluated)[::-1]
            for idx in sorted_indices[:min(3, len(sorted_indices))]:
                if len(population) < population_size // 2:
                    population.append(self.configs_evaluated[idx].copy())
        
        # Fill rest with random configurations
        while len(population) < population_size:
            config = {h: random.choice(VOLTAGE_OPTIONS) for h in MODIFIABLE_HEATERS}
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:
                    config[h] = 0.01
            population.append(config)
        
        # Evolutionary loop
        for generation in range(generations):
            print(f"Generation {generation+1}/{generations}")
            
            # Evaluate all members
            scores = []
            for i, config in enumerate(population):
                # Use surrogate model if available, otherwise evaluate on hardware
                if hasattr(self, 'model') and self.model is not None:
                    score = self.predict_score(config)
                    scores.append(score)
                else:
                    score, _ = self.evaluate_configuration(config)
                    scores.append(score)
            
            # Select parents - tournament selection
            parents = []
            for _ in range(population_size):
                # Select 3 random individuals
                candidates = random.sample(range(population_size), 3)
                # Choose the best one
                winner = max(candidates, key=lambda idx: scores[idx])
                parents.append(population[winner])
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individual
            best_idx = np.argmax(scores)
            new_population.append(population[best_idx].copy())
            
            # Fill rest with offspring
            while len(new_population) < population_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                child = {}
                for h in MODIFIABLE_HEATERS:
                    # 70% chance to inherit from better parent, 30% from random parent
                    if random.random() < 0.7:
                        score1 = self.predict_score({h: parent1[h]}) if hasattr(self, 'model') else 0
                        score2 = self.predict_score({h: parent2[h]}) if hasattr(self, 'model') else 0
                        child[h] = parent1[h] if score1 > score2 else parent2[h]
                    else:
                        child[h] = random.choice([parent1[h], parent2[h]])
                
                # Mutation
                for h in MODIFIABLE_HEATERS:
                    if random.random() < 0.2:  # 20% mutation rate
                        current = child[h]
                        # Prefer nearby values in the discrete space
                        current_idx = VOLTAGE_OPTIONS.index(current) if current in VOLTAGE_OPTIONS else 0
                        max_step = min(2, len(VOLTAGE_OPTIONS) - 1)  # Maximum step size in voltage options
                        step = random.randint(-max_step, max_step)
                        new_idx = max(0, min(len(VOLTAGE_OPTIONS) - 1, current_idx + step))
                        child[h] = VOLTAGE_OPTIONS[new_idx]
                
                # Add fixed layer values
                for h in FIXED_FIRST_LAYER:
                    if h not in INPUT_HEATERS:
                        child[h] = 0.01
                
                new_population.append(child)
            
            # Replace population
            population = new_population
        
        # Evaluate most promising configurations on actual hardware
        final_candidates = []
        
        # Predict scores for all final population members
        if hasattr(self, 'model') and self.model is not None:
            predicted_scores = [self.predict_score(config) for config in population]
            sorted_indices = np.argsort(predicted_scores)[::-1]
            
            # Take top 3 candidates
            for idx in sorted_indices[:3]:
                final_candidates.append(population[idx])
        else:
            final_candidates = population[:3]
        
        # Evaluate candidates on hardware
        for candidate in final_candidates:
            self.evaluate_configuration(candidate)
    
    def adaptive_grid_search(self, n_iterations=5):
        """Perform adaptive grid search over promising dimensions"""
        print("\nPerforming adaptive grid search...")
        
        if not self.best_config:
            print("No best configuration available yet.")
            return
        
        base_config = self.best_config.copy()
        
        # Determine most important features if model is available
        important_features = None
        if hasattr(self, 'model') and self.model is not None:
            importances = self.model.feature_importances_
            # Get indices of top 5 most important features
            important_indices = np.argsort(importances)[::-1][:5]
            important_features = [MODIFIABLE_HEATERS[i] for i in important_indices]
            print(f"Most important heaters: {important_features}")
        else:
            # If no model, choose 5 random heaters
            important_features = random.sample(MODIFIABLE_HEATERS, 5)
        
        # Grid search over important features
        for iteration in range(n_iterations):
            print(f"Grid search iteration {iteration+1}/{n_iterations}")
            
            if iteration > 0:
                # Update important features based on recent improvements
                if hasattr(self, 'model') and self.model is not None:
                    self.train_surrogate_model()
                    importances = self.model.feature_importances_
                    important_indices = np.argsort(importances)[::-1][:5]
                    important_features = [MODIFIABLE_HEATERS[i] for i in important_indices]
                    print(f"Updated important heaters: {important_features}")
            
            # For each important feature, try different values
            for feature in important_features:
                current_value = base_config[feature]
                best_value = current_value
                best_score = float('-inf')
                
                # Try different values
                for value in [v for v in VOLTAGE_OPTIONS if v != current_value]:
                    test_config = base_config.copy()
                    test_config[feature] = value
                    
                    score, _ = self.evaluate_configuration(test_config)
                    
                    if score > best_score:
                        best_score = score
                        best_value = value
                
                # Update base config with best value
                base_config[feature] = best_value
        
    def differential_evolution_step(self):
        """Run differential evolution on promising regions"""
        print("\nRunning differential evolution optimization...")
        
        if len(self.configs_evaluated) < 10:
            print("Not enough data for differential evolution yet.")
            return
        
        # Train surrogate model
        self.train_surrogate_model()
        
        # Define bounds for optimization
        bounds = [(0, len(VOLTAGE_OPTIONS)-1) for _ in range(len(MODIFIABLE_HEATERS))]
        
        # Objective function for differential evolution
        def objective(indices):
            # Convert indices to voltages
            voltages = [VOLTAGE_OPTIONS[int(round(idx))] for idx in indices]
            config = self.vector_to_config(voltages)
            
            # Use surrogate model for faster evaluation
            return -self.predict_score(config)  # Negative because we're minimizing
        
        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            popsize=8,
            maxiter=10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            strategy='best1bin',
            disp=True
        )
        
        # Convert result to configuration
        indices = [int(round(idx)) for idx in result.x]
        voltages = [VOLTAGE_OPTIONS[idx] for idx in indices]
        best_config = self.vector_to_config(voltages)
        
        # Evaluate on actual hardware
        print("Evaluating DE result on hardware...")
        self.evaluate_configuration(best_config)
    
    def local_search(self, iterations=10):
        """Perform local search around best configuration"""
        print("\nPerforming local search around best configuration...")
        
        if not self.best_config:
            print("No best configuration available yet.")
            return
        
        current_config = self.best_config.copy()
        current_score, _ = self.evaluate_configuration(current_config)
        
        for iteration in range(iterations):
            print(f"Local search iteration {iteration+1}/{iterations}")
            improved = False
            
            # Randomly select heaters to optimize
            heaters = random.sample(MODIFIABLE_HEATERS, min(5, len(MODIFIABLE_HEATERS)))
            
            for heater in heaters:
                current_value = current_config[heater]
                
                # Try nearby voltage values
                current_idx = VOLTAGE_OPTIONS.index(current_value)
                neighbor_indices = [
                    max(0, current_idx - 1),
                    min(len(VOLTAGE_OPTIONS) - 1, current_idx + 1)
                ]
                
                for idx in neighbor_indices:
                    if idx == current_idx:
                        continue
                    
                    test_config = current_config.copy()
                    test_config[heater] = VOLTAGE_OPTIONS[idx]
                    
                    score, _ = self.evaluate_configuration(test_config)
                    
                    if score > current_score:
                        current_score = score
                        current_config = test_config.copy()
                        improved = True
                        print(f"  Improved score to: {current_score:.2f}")
                        break
                
                if improved:
                    break
            
            if not improved:
                print("  No further improvements found")
                break
        
        return current_config, current_score
    
    def optimize(self):
        """Run multi-stage optimization for OR gate"""
        print("Starting OR gate optimization...")
        
        # Phase 1: Initial exploration
        self.initial_sampling(n_samples=10)
        
        # Phase 2: Train initial surrogate model
        self.train_surrogate_model()
        
        # Phase 3: Evolutionary optimization
        self.evolution_step(population_size=8, generations=3)
        
        # Phase 4: Retrain surrogate model
        self.train_surrogate_model()
        
        # Phase 5: Adaptive grid search
        self.adaptive_grid_search(n_iterations=3)
        
        # Phase 6: Differential evolution
        self.differential_evolution_step()
        
        # Phase 7: Final local search
        final_config, final_score = self.local_search(iterations=10)
        
        print("\nOptimization complete!")
        print(f"Best score: {final_score:.2f}")
        
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
            expected_high = input_state != (4.9, 4.9)  # Only (High,High) should be LOW
            is_correct = is_high == expected_high

            if not is_correct:
                all_correct = False

            print(f"\nInputs (A, B): {input_state}")
            print(f"NAND Output: {output_value:.4f}V")
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
    
    def format_config(self, config):
            """Format configuration values for display and ensure all heaters 0-39 are included"""
            # Start with a complete range of heaters
            complete_config = {i: 0.0 for i in range(40)}
            
            # Add all values from the optimization config
            for heater, value in config.items():
                complete_config[heater] = value
                
            # Make sure fixed first layer heaters are set correctly
            for h in FIXED_FIRST_LAYER:
                if h not in INPUT_HEATERS:  # Don't override input heaters
                    complete_config[h] = 0.01
                    
            # Format all values to 2 decimal places
            return {k: round(float(v), 2) for k, v in complete_config.items()}



def main():
    optimizer = ORGateOptimizer()
    
    try:
        # Run optimization
        best_config, best_score = optimizer.optimize()
        
        # Print final heater configuration
        print("\nFinal Heater Configuration:")

        # for heater in sorted(best_config.keys()):
        #     print(f"Heater {heater}: {best_config[heater]:.2f}V")
        print(best_score)
        clean_config = optimizer.format_config(best_config)
        print(clean_config)
        
        # Test final configuration
        optimizer.test_final_configuration(best_config)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    main()