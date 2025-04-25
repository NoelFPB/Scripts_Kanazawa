import serial
import time
import pyvisa
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution
import random
from itertools import product


# Not working yet, pretty hard behaviour to achieve

# Serial port configuration 
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# === DEMUX CONFIGURATION - MODIFY THIS SECTION ===
# Demultiplexer setup 
# A demultiplexer routes 1 input to 1 of several outputs based on select lines
DATA_INPUT_HEATER = 36     # Data input 
SELECT_HEATERS = [37, 38]  # Select lines (S0, S1)

# For a 1-to-4 demux with 2 select lines:
# S1 S0 | Active Output
# 0  0  | Output 0 (Channel 1)
# 0  1  | Output 1 (Channel 2)
# 1  0  | Output 2 (Channel 3)
# 1  1  | Output 3 (Channel 4)

# Input/select line voltage values representing LOW and HIGH states
LOW_VOLTAGE = 0.1    # Voltage representing logical LOW
HIGH_VOLTAGE = 4.9   # Voltage representing logical HIGH

# Output voltage thresholds
LOW_THRESHOLD = 2.5   # Outputs below this are considered LOW
OPTIMAL_LOW = 2     # Target for LOW output

HIGH_THRESHOLD = 3.5  # Outputs above this are considered HIGH
OPTIMAL_HIGH = 4    # Target for HIGH output

# Heater configuration
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in [DATA_INPUT_HEATER] + SELECT_HEATERS]

# Test configurations for demux
# Format: (data_input, select_s0, select_s1)
TEST_CONFIGURATIONS = [
    (LOW_VOLTAGE, LOW_VOLTAGE, LOW_VOLTAGE),   # Low data, select 00 (output 0 should be low)
    (HIGH_VOLTAGE, LOW_VOLTAGE, LOW_VOLTAGE),  # High data, select 00 (output 0 should be high)
    (LOW_VOLTAGE, HIGH_VOLTAGE, LOW_VOLTAGE),  # Low data, select 01 (output 1 should be low)
    (HIGH_VOLTAGE, HIGH_VOLTAGE, LOW_VOLTAGE), # High data, select 01 (output 1 should be high)
    (LOW_VOLTAGE, LOW_VOLTAGE, HIGH_VOLTAGE),  # Low data, select 10 (output 2 should be low)
    (HIGH_VOLTAGE, LOW_VOLTAGE, HIGH_VOLTAGE), # High data, select 10 (output 2 should be high)
    (LOW_VOLTAGE, HIGH_VOLTAGE, HIGH_VOLTAGE), # Low data, select 11 (output 3 should be low)
    (HIGH_VOLTAGE, HIGH_VOLTAGE, HIGH_VOLTAGE) # High data, select 11 (output 3 should be high)
]

# Voltage options for discretization
def create_voltage_options(start=0, end=4.9, step=0.1):
    options = []
    current = start
    options.append(0.1)
    
    while current <= end:
        options.append(round(current, 2))  # Round to 2 decimal places
        current += step
    
    # Add the final value (4.9) if it's not already included
    if end not in options:
        options.append(end)
    if 0 in options:
        options.remove(0)
        
    return options

# Voltage options with different granularity
VOLTAGE_OPTIONS = create_voltage_options()
VOLTAGE_OPTIONS_GRID = create_voltage_options(start=0, end=4.9, step=0.5)
print(VOLTAGE_OPTIONS)


class DemuxOptimizer:
    """
    Optimize a physical demultiplexer using ensemble learning and evolutionary strategies.
    """
    def __init__(self):
        print("Initializing demultiplexer optimization...")
        
        # Initialize hardware connections
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Initialize Random Forest model for surrogate modeling
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
        """Initialize oscilloscope for demux output measurement"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000

        # Setup all 4 channels for demux output measurement
        for channel_num in range(1, 5):
            scope.write(f':CHANnel{channel_num}:DISPlay ON')
            scope.write(f':CHANnel{channel_num}:SCALe 2')
            scope.write(f':CHANnel{channel_num}:OFFSet -6')
            
        return scope
    
    def send_heater_values(self, config):
        """Send heater voltage configuration to hardware"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.serial.write(voltage_message.encode())
        self.serial.flush()
        time.sleep(0.01)  # Small delay to ensure message is sent
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
    
    def measure_outputs(self):
        """Measure all 4 demux output voltages from oscilloscope"""
        try:
            outputs = []
            for channel in range(1, 5):
                value = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                outputs.append(round(value, 5))
            return outputs
        except Exception as e:
            print(f"Measurement error: {e}")
            return [None] * 4
    
    def config_to_vector(self, config):
        """Convert configuration dictionary to feature vector for model"""
        return np.array([config.get(h, 0.1) for h in MODIFIABLE_HEATERS])
    
    def vector_to_config(self, vector):
        """Convert feature vector to configuration dictionary"""
        config = {h: vector[i] for i, h in enumerate(MODIFIABLE_HEATERS)}
        
        # Add fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in [DATA_INPUT_HEATER] + SELECT_HEATERS:
                config[h] = 0.01
                
        return config
    
    def evaluate_configuration(self, config):
        """
        Evaluate demultiplexer configuration based on how well it routes inputs
        to the appropriate output based on select lines.
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
        
        # Test each input/select combination
        for test_config in TEST_CONFIGURATIONS:
            data_value, select_s0, select_s1 = test_config
            
            # Determine which output should be active based on select lines
            expected_output_idx = (1 if select_s0 > LOW_THRESHOLD else 0) + 2 * (1 if select_s1 > LOW_THRESHOLD else 0)
            
            # Set up current test configuration
            current_config = config.copy()
            current_config[DATA_INPUT_HEATER] = data_value
            current_config[SELECT_HEATERS[0]] = select_s0
            current_config[SELECT_HEATERS[1]] = select_s1
            
            # Send configuration to hardware
            self.send_heater_values(current_config)
            time.sleep(0.20)  # Wait for the system to stabilize
            
            # Measure all outputs
            outputs = self.measure_outputs()
            if None in outputs:
                return -1000, []  # Return a large negative score on error
            
            # Score based on whether the correct output is active
            # and whether inactive outputs are LOW
            expected_values = [LOW_VOLTAGE] * 4
            expected_values[expected_output_idx] = data_value  # The selected output should match the data input
            
            for i, (output, expected) in enumerate(zip(outputs, expected_values)):
                if expected > LOW_THRESHOLD:  # Should be HIGH
                    if output > OPTIMAL_HIGH and output < 6:
                        # Stronger reward for cleaner HIGH signal on the correct output
                        total_score += 40 + (5 * output)
                    elif output > HIGH_THRESHOLD and output < 6:
                        # Acceptable but not optimal
                        total_score += 20 + (5 * output)
                    else:
                        # Severe penalty for incorrect behavior
                        total_score -= 50 + ((5.0 - output) * (5.0 - output))
                else:  # Should be LOW
                    if output < OPTIMAL_LOW:
                        # Stronger reward for cleaner LOW signal on inactive outputs
                        total_score += 40 + (20 * np.exp(-output))
                    elif output < LOW_THRESHOLD:
                        # Acceptable but not optimal
                        total_score += 20 + (10 * (1.5 - output))
                    else:
                        # Severe penalty for incorrect behavior
                        total_score -= 50 + (output * output)
            
            # Additional bonus for separation between active and inactive outputs
            active_output = outputs[expected_output_idx]
            inactive_outputs = [out for i, out in enumerate(outputs) if i != expected_output_idx]
            
            if active_output > LOW_THRESHOLD and max(inactive_outputs) < LOW_THRESHOLD:
                # Bonus for clean separation
                separation = active_output - max(inactive_outputs)
                total_score += separation * 10  # Reward separation
            
            # Record results
            results.append({
                'data_input': data_value,
                'select_lines': (select_s0, select_s1),
                'outputs': outputs,
                'expected_active': expected_output_idx,
                'correctly_routed': (
                    outputs[expected_output_idx] > HIGH_THRESHOLD if data_value > LOW_THRESHOLD 
                    else outputs[expected_output_idx] < LOW_THRESHOLD
                )
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
        """Initial sampling using Latin Hypercube Sampling (LHS) for better space coverage"""
        from scipy.stats import qmc  # More standard library for experimental design
        
        print(f"Performing Latin Hypercube initial sampling with {n_samples} configurations...")
        
        # Zero configuration (baseline)
        zero_config = {h: 0.1 for h in MODIFIABLE_HEATERS}
        for h in FIXED_FIRST_LAYER:
            if h not in [DATA_INPUT_HEATER] + SELECT_HEATERS:
                zero_config[h] = 0.01
        
        score, _ = self.evaluate_configuration(zero_config)
        print(f"Zero configuration: Score = {score:.2f}")
        
        # Track configs we've evaluated
        tried_configs = set()
        tried_configs.add(tuple(sorted(zero_config.items())))
        
        # Number of dimensions = number of modifiable heaters
        n_dim = len(MODIFIABLE_HEATERS)
        
        # Generate Latin Hypercube samples for better coverage of the search space
        sampler = qmc.LatinHypercube(d=n_dim, seed=42)
        samples = sampler.random(n=n_samples-1)  # -1 because we already did the zero config
        
        # Convert uniform samples [0,1] to voltage range
        voltage_min = min(VOLTAGE_OPTIONS)
        voltage_max = max(VOLTAGE_OPTIONS)
        
        # Function to find closest allowed voltage
        def closest_voltage(v):
            return min(VOLTAGE_OPTIONS, key=lambda x: abs(x - v))
        
        # Evaluate each sample
        for i, sample in enumerate(samples):
            # Convert unit hypercube to voltage range
            voltages = voltage_min + sample * (voltage_max - voltage_min)
            
            # Create configuration (with discretization to allowed voltages)
            config = {}
            for j, h in enumerate(MODIFIABLE_HEATERS):
                config[h] = closest_voltage(voltages[j])
            
            # Add fixed first layer values
            for h in FIXED_FIRST_LAYER:
                if h not in [DATA_INPUT_HEATER] + SELECT_HEATERS:
                    config[h] = 0.01
            
            # Check if configuration is unique
            config_tuple = tuple(sorted(config.items()))
            if config_tuple in tried_configs:
                # Simple perturbation if duplicate
                for h in random.sample(MODIFIABLE_HEATERS, 3):  # Perturb 3 random heaters
                    config[h] = random.choice(VOLTAGE_OPTIONS)
                config_tuple = tuple(sorted(config.items()))
            
            tried_configs.add(config_tuple)
            # Evaluate configuration
            score, _ = self.evaluate_configuration(config)
            print(f"Initial configuration {i+1}/{n_samples-1}: Score = {score:.2f}")
        
        # Optional: Add some fully random samples for exploration
        n_random = max(0, (n_samples-1) // 4)  # 25% fully random samples
        
        for i in range(n_random):
            config = {h: random.choice(VOLTAGE_OPTIONS) for h in MODIFIABLE_HEATERS}
            for h in FIXED_FIRST_LAYER:
                if h not in [DATA_INPUT_HEATER] + SELECT_HEATERS:
                    config[h] = 0.01
            
            config_tuple = tuple(sorted(config.items()))
            if config_tuple not in tried_configs:
                tried_configs.add(config_tuple)
                score, _ = self.evaluate_configuration(config)
                print(f"Random configuration {i+1}/{n_random}: Score = {score:.2f}")

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
                if h not in [DATA_INPUT_HEATER] + SELECT_HEATERS:
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
                    if h not in [DATA_INPUT_HEATER] + SELECT_HEATERS:
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
                for value in [v for v in VOLTAGE_OPTIONS_GRID if v != current_value]:
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
        """Run multi-stage optimization for demultiplexer"""
        print("Starting demultiplexer optimization...")
        
        # Phase 1: Initial exploration
        self.initial_sampling(n_samples=200)
        
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
        """Test and print performance of the optimized demultiplexer"""
        print("\nTesting final demultiplexer configuration:")
        all_correct = True
        
        for test_config in TEST_CONFIGURATIONS:
            data_value, select_s0, select_s1 = test_config
            
            # Determine which output should be active based on select lines
            expected_output_idx = (1 if select_s0 > LOW_THRESHOLD else 0) + 2 * (1 if select_s1 > LOW_THRESHOLD else 0)
            
            # Set up current test configuration
            current_config = config.copy()
            current_config[DATA_INPUT_HEATER] = data_value
            current_config[SELECT_HEATERS[0]] = select_s0
            current_config[SELECT_HEATERS[1]] = select_s1
            
            # Send configuration to hardware
            self.send_heater_values(current_config)
            time.sleep(0.25)
            
            # Measure all outputs
            outputs = self.measure_outputs()
            
            # Check if the correct output is active and others are inactive
            expected_active_output = data_value  # Expected output matches input value
            
            # Define what "active" means based on data input value
            if data_value > LOW_THRESHOLD:  # HIGH input
                expected_active = expected_output_idx
                is_correct = (outputs[expected_active] > HIGH_THRESHOLD and 
                             all(out < LOW_THRESHOLD for i, out in enumerate(outputs) if i != expected_active))
            else:  # LOW input - all outputs should be LOW
                is_correct = all(out < LOW_THRESHOLD for out in outputs)
            
            if not is_correct:
                all_correct = False
            
            # Determine status of each output (HIGH/LOW)
            output_states = ["HIGH" if out > HIGH_THRESHOLD else "LOW" for out in outputs]
            
            print(f"\nTest Case:")
            print(f"  Data Input: {'HIGH' if data_value > HIGH_THRESHOLD else 'LOW'} ({data_value}V)")
            print(f"  Select Lines (S0, S1): {select_s0}V, {select_s1}V")
            print(f"  Select Code: {int(select_s0 > HIGH_THRESHOLD)}{int(select_s1 > HIGH_THRESHOLD)}")
            print(f"  Expected Active Output: Channel {expected_output_idx+1}")
            print(f"  Measured Outputs:")
            for i, (out, state) in enumerate(zip(outputs, output_states)):
                print(f"    Channel {i+1}: {out:.4f}V ({state})")
            print(f"  Correct Routing: {'Yes' if is_correct else 'No'}")
        
        if all_correct:
            print("\nSuccess! The demultiplexer is working correctly for all input combinations.")
        else:
            print("\nThe demultiplexer is not working perfectly for all input combinations.")
        
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
            if h not in [DATA_INPUT_HEATER] + SELECT_HEATERS:  # Don't override input/select heaters
                complete_config[h] = 0.01
                
        # Format all values to 2 decimal places
        return {k: round(float(v), 2) for k, v in complete_config.items()}
    

def main():
    # Create demultiplexer optimizer
    optimizer = DemuxOptimizer()

    try:
        # Run optimization
        best_config, best_score = optimizer.optimize()
        
        # Print final heater configuration
        print("\nFinal Demultiplexer Heater Configuration:")
        print(f"Best score: {best_score}")
        
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