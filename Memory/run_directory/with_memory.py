import serial
import time
import pyvisa
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution
import random
from scipy.stats import qmc
import os
import pickle
import datetime
import json

# Serial port configuration 
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# === DECODER CONFIGURATION - MODIFY THIS SECTION ===
# Decoder setup 
# A decoder converts binary input(s) into one-hot outputs with only one active output
INPUT_PINS = [36, 37]  # Input pins (A, B)

# For a 2-to-4 decoder with 2 inputs:
# A B | Active Output
# 0 0 | Output 0 
# 0 1 | Output 1 
# 1 0 | Output 2 
# 1 1 | Output 3 

# Input voltage values representing LOW and HIGH states
LOW_VOLTAGE = 0.1    # Voltage representing logical LOW
HIGH_VOLTAGE = 4.9   # Voltage representing logical HIGH

# Output voltage thresholds
LOW_THRESHOLD = 2   # Outputs below this are considered LOW
OPTIMAL_LOW = 1.5     # Target for LOW output

HIGH_THRESHOLD = 3.5  # Outputs above this are considered HIGH  
OPTIMAL_HIGH = 4    # Target for HIGH output

# Heater configuration
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in INPUT_PINS]

# Test configurations for decoder
# Format: (input_a, input_b)
TEST_CONFIGURATIONS = [
    (LOW_VOLTAGE, LOW_VOLTAGE),    # Input 00 (output 0 should be high, others low)
    (LOW_VOLTAGE, HIGH_VOLTAGE),   # Input 01 (output 1 should be high, others low)
    (HIGH_VOLTAGE, LOW_VOLTAGE),   # Input 10 (output 2 should be high, others low)
    (HIGH_VOLTAGE, HIGH_VOLTAGE)   # Input 11 (output 3 should be high, others low)
]

# File paths for persistence
DATA_DIR = "./memory/decoder_data"
MODEL_PATH = os.path.join(DATA_DIR, "surrogate_model.pkl")
HISTORY_PATH = os.path.join(DATA_DIR, "evaluation_history.pkl")
BEST_CONFIG_PATH = os.path.join(DATA_DIR, "best_configs.json")

# Voltage options for discretization
def create_voltage_options(start=0, end=4.9, step=0.05):
    options = []
    current = start
    
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
VOLTAGE_OPTIONS_GRID = create_voltage_options(start=0, end=4.9, step=0.05)


class DecoderOptimizer:
    """
    Optimize a physical decoder using ensemble learning and evolutionary strategies.
    With persistence capabilities to save and load model data between runs.
    """
    def __init__(self, load_previous=True):
        print("Initializing decoder optimization...")
        print(VOLTAGE_OPTIONS)
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
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
        
        # Best configurations found
        self.best_config = None
        self.best_score = float('-inf')
        self.best_configs_history = []
        
        # Load previous data if available and requested
        if load_previous:
            self.load_data()
            
    def _init_scope(self):
        """Initialize oscilloscope for decoder output measurement"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000

        # Setup all 4 channels for decoder output measurement
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
        """Measure all 4 decoder output voltages from oscilloscope"""
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
            if h not in INPUT_PINS:
                config[h] = 0.01
                
        return config
    
    def evaluate_configuration(self, config):
        """Evaluate decoder configuration with consistency metrics."""
        # Check if already evaluated
        config_key = tuple(config.get(h, 0.1) for h in sorted(config.keys()))
        for i, past_config in enumerate(self.configs_evaluated):
            past_key = tuple(past_config.get(h, 0.1) for h in sorted(past_config.keys()))
            if config_key == past_key:
                return self.scores_evaluated[i], None
        
        # Component weights (must sum to 1.0)
        WEIGHTS = {
            'high_state': 0.2,       # HIGH output performance
            'low_state': 0.2,        # LOW output performance
            'high_consistency': 0.2, # Consistency across HIGH outputs
            'low_consistency': 0.2,  # Consistency across LOW outputs
            'separation': 0.2        # Separation between HIGH/LOW
        }
        
        # Initialize scores
        component_scores = {k: 0.0 for k in WEIGHTS}
        results = []
        
        # Collection of all high and low outputs for consistency calculation
        all_high_outputs = []  # Active outputs (should be HIGH)
        all_low_outputs = []   # Inactive outputs (should be LOW)
        
        # Test each input combination
        for test_config in TEST_CONFIGURATIONS:
            input_a, input_b = test_config
            
            # Determine expected active output
            expected_output_idx = (1 if input_b > LOW_THRESHOLD else 0) + 2 * (1 if input_a > LOW_THRESHOLD else 0)
            
            # Configure and measure
            current_config = config.copy()
            current_config[INPUT_PINS[0]] = input_a
            current_config[INPUT_PINS[1]] = input_b
            self.send_heater_values(current_config)
            time.sleep(0.20)
            
            outputs = self.measure_outputs()
            if None in outputs:
                return -1000, []
            
            # Get active and inactive outputs
            active_output = outputs[expected_output_idx]
            inactive_outputs = [out for i, out in enumerate(outputs) if i != expected_output_idx]
            
            # Collect outputs for consistency calculation
            all_high_outputs.append(active_output)
            all_low_outputs.extend(inactive_outputs)
            
            # Score HIGH state (simplified)
            if active_output >= OPTIMAL_HIGH:
                high_score = 1.0
            elif active_output >= HIGH_THRESHOLD:
                high_score = 0.5 + 0.5 * ((active_output - HIGH_THRESHOLD) / (OPTIMAL_HIGH - HIGH_THRESHOLD))
            else:
                high_score = 0.5 * (active_output / HIGH_THRESHOLD)
            
            # Score LOW states (simplified)
            low_scores = []
            for out in inactive_outputs:
                if out <= OPTIMAL_LOW:
                    low_scores.append(1.0)
                elif out <= LOW_THRESHOLD:
                    low_scores.append(0.5 + 0.5 * ((LOW_THRESHOLD - out) / (LOW_THRESHOLD - OPTIMAL_LOW)))
                else:
                    low_scores.append(0.5 * (1 - min(1, (out - LOW_THRESHOLD) / (HIGH_THRESHOLD - LOW_THRESHOLD))))
            
            avg_low_score = sum(low_scores) / len(low_scores)
            
            # Score separation
            max_inactive = max(inactive_outputs)
            separation_score = min(1.0, max(0, (active_output - max_inactive) / (OPTIMAL_HIGH - OPTIMAL_LOW)))
            
            # Update component scores
            component_scores['high_state'] += high_score / len(TEST_CONFIGURATIONS)
            component_scores['low_state'] += avg_low_score / len(TEST_CONFIGURATIONS)
            component_scores['separation'] += separation_score / len(TEST_CONFIGURATIONS)
            
            # Record binary success
            correctly_decoded = (active_output > HIGH_THRESHOLD and all(out < LOW_THRESHOLD for out in inactive_outputs))
            
            # Store results for this test
            results.append({
                'inputs': (input_a, input_b),
                'expected_active': expected_output_idx,
                'outputs': outputs,
                'correctly_decoded': correctly_decoded
            })
        
        # Calculate consistency scores for HIGH outputs
        if len(all_high_outputs) > 1:
            high_avg = sum(all_high_outputs) / len(all_high_outputs)
            high_variance = sum((o - high_avg)**2 for o in all_high_outputs) / len(all_high_outputs)
            # Convert variance to consistency score (lower variance = higher consistency)
            component_scores['high_consistency'] = 1.0 / (1.0 + (high_variance / 0.25))
        else:
            # This should never happen for a decoder with 4 test cases, but included for robustness
            component_scores['high_consistency'] = 1.0
        
        # Calculate consistency scores for LOW outputs
        if len(all_low_outputs) > 1:
            low_avg = sum(all_low_outputs) / len(all_low_outputs)
            low_variance = sum((o - low_avg)**2 for o in all_low_outputs) / len(all_low_outputs)
            # Convert variance to consistency score
            component_scores['low_consistency'] = 1.0 / (1.0 + (low_variance / 0.25))
        else:
            # This should never happen for a decoder with 4 test cases, but included for robustness
            component_scores['low_consistency'] = 1.0
        
        # Calculate final score (0-100)
        final_score = sum(component_scores[k] * WEIGHTS[k] * 100 for k in WEIGHTS)
        
        # Add success bonus
        success_count = sum(1 for r in results if r['correctly_decoded'])
        if success_count == len(TEST_CONFIGURATIONS):
            final_score += 20  # Bonus for perfect decoding
        
        # Add to history
        self.configs_evaluated.append(config.copy())
        self.scores_evaluated.append(final_score)
        
        # Update best if improved
        if final_score > self.best_score:
            self.best_score = final_score
            self.best_config = config.copy()
            
            # Add to best configs history with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.best_configs_history.append({
                "timestamp": timestamp,
                "score": final_score,
                "config": self.format_config(config),
                "success_count": success_count,
                "component_scores": component_scores
            })
            
            print(f"\nNew best: Score={final_score:.2f}, Success={success_count}/{len(TEST_CONFIGURATIONS)}")
            print(f"Components: HIGH={component_scores['high_state']:.2f}, "
                f"LOW={component_scores['low_state']:.2f}, "
                f"HIGH_CONS={component_scores['high_consistency']:.2f}, "
                f"LOW_CONS={component_scores['low_consistency']:.2f}, "
                f"SEP={component_scores['separation']:.2f}")
            
            if all_high_outputs:
                print(f"HIGH outputs: avg={sum(all_high_outputs)/len(all_high_outputs):.2f}V, "
                    f"min={min(all_high_outputs):.2f}V, max={max(all_high_outputs):.2f}V")
            if all_low_outputs:
                print(f"LOW outputs: avg={sum(all_low_outputs)/len(all_low_outputs):.2f}V, "
                    f"min={min(all_low_outputs):.2f}V, max={max(all_low_outputs):.2f}V")
            
            # Save data after each new best configuration
            self.save_data()
        
        return final_score, results

    def save_data(self):
        """Save model, evaluation history, and best configurations to disk"""
        print("Saving data to disk...")
        
        # Save surrogate model
        if hasattr(self, 'model') and self.model is not None:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save evaluation history
        with open(HISTORY_PATH, 'wb') as f:
            history_data = {
                'configs_evaluated': self.configs_evaluated,
                'scores_evaluated': self.scores_evaluated,
                'best_config': self.best_config,
                'best_score': self.best_score
            }
            pickle.dump(history_data, f)
        
        # Save best configs history as JSON (more readable)
        with open(BEST_CONFIG_PATH, 'w') as f:
            json.dump(self.best_configs_history, f, indent=2)
            
        print(f"Data saved. Total configurations evaluated: {len(self.configs_evaluated)}")
    
    def load_data(self):
        """Load model and evaluation history from disk if available"""
        # Check if files exist
        model_exists = os.path.isfile(MODEL_PATH)
        history_exists = os.path.isfile(HISTORY_PATH)
        best_configs_exists = os.path.isfile(BEST_CONFIG_PATH)
        
        if not (model_exists and history_exists):
            print("No previous data found. Starting fresh.")
            return False
        
        try:
            # Load surrogate model
            if model_exists:
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                print("Loaded surrogate model.")
            
            # Load evaluation history
            if history_exists:
                with open(HISTORY_PATH, 'rb') as f:
                    history_data = pickle.load(f)
                    self.configs_evaluated = history_data['configs_evaluated']
                    self.scores_evaluated = history_data['scores_evaluated']
                    self.best_config = history_data['best_config']
                    self.best_score = history_data['best_score']
                print(f"Loaded evaluation history. {len(self.configs_evaluated)} configurations found.")
            
            # Load best configs history
            if best_configs_exists:
                with open(BEST_CONFIG_PATH, 'r') as f:
                    self.best_configs_history = json.load(f)
                print(f"Loaded {len(self.best_configs_history)} best configurations.")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Starting fresh.")
            return False

    def initial_sampling(self, n_samples=10):
        """Initial sampling using Latin Hypercube Sampling (LHS) for better space coverage"""
        print(f"Performing Latin Hypercube initial sampling with {n_samples} configurations...")
        
        # If we already have data, reduce the number of initial samples
        if len(self.configs_evaluated) > 0:
            original_n = n_samples
            n_samples = max(5, n_samples // 2)  # Reduce but ensure at least 5
            print(f"Using reduced sample size ({n_samples} instead of {original_n}) due to existing data.")
        
        # Zero configuration (baseline)
        zero_config = {h: 0.1 for h in MODIFIABLE_HEATERS}
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_PINS:
                zero_config[h] = 0.01
        
        score, _ = self.evaluate_configuration(zero_config)
        print(f"Zero configuration: Score = {score:.2f}")
        
        # Track configs we've evaluated
        tried_configs = set()
        tried_configs.add(tuple(sorted(zero_config.items())))
        
        # Number of dimensions = number of modifiable heaters
        n_dim = len(MODIFIABLE_HEATERS)
        
        # Generate Latin Hypercube samples for better coverage of the search space
        sampler = qmc.LatinHypercube(d=n_dim, seed=2)
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
                if h not in INPUT_PINS:
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
                if h not in INPUT_PINS:
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
                if h not in INPUT_PINS:
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
                    if h not in INPUT_PINS:
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
    
    def adaptive_grid_search(self, n_iterations=3, feature_count=3, grid_points=5):
        """Perform faster adaptive grid search over promising dimensions"""
        print("\nPerforming faster adaptive grid search...")
        
        if not self.best_config:
            print("No best configuration available yet.")
            return
        
        base_config = self.best_config.copy()
        
        # Determine most important features if model is available
        important_features = None
        if hasattr(self, 'model') and self.model is not None:
            importances = self.model.feature_importances_
            # Get indices of top N most important features (reduced from 5)
            important_indices = np.argsort(importances)[::-1][:feature_count]
            important_features = [MODIFIABLE_HEATERS[i] for i in important_indices]
            print(f"Most important heaters: {important_features}")
        else:
            # If no model, choose fewer random heaters
            important_features = random.sample(MODIFIABLE_HEATERS, feature_count)
        
        # Create a smaller, more targeted grid
        def create_targeted_grid(current_value, min_val=0.1, max_val=4.9, n_points=grid_points):
            """Create a small grid centered around the current value"""
            # Make sure current value is in the grid
            grid = [current_value]
            
            # Determine step size to create n_points total
            range_size = max_val - min_val
            step = range_size / (n_points - 1)
            
            # Calculate potential grid points
            potential_points = [min_val + i * step for i in range(n_points)]
            
            # Add points not already in the grid
            for point in potential_points:
                rounded_point = round(point, 1)  # Round to match VOLTAGE_OPTIONS precision
                if rounded_point != current_value and rounded_point >= min_val and rounded_point <= max_val:
                    grid.append(rounded_point)
            
            return grid
        
        # Grid search over important features (reduced iterations)
        for iteration in range(n_iterations):
            print(f"Grid search iteration {iteration+1}/{n_iterations}")
            
            if iteration > 0:
                # Update important features based on recent improvements
                if hasattr(self, 'model') and self.model is not None:
                    self.train_surrogate_model()
                    importances = self.model.feature_importances_
                    important_indices = np.argsort(importances)[::-1][:feature_count]
                    important_features = [MODIFIABLE_HEATERS[i] for i in important_indices]
                    print(f"Updated important heaters: {important_features}")
            
            # For each important feature, try a smaller set of values
            for feature in important_features:
                current_value = base_config[feature]
                best_value = current_value
                best_score = float('-inf')
                
                # Get targeted grid values instead of using all VOLTAGE_OPTIONS_GRID
                test_values = create_targeted_grid(current_value)
                print(f"  Testing heater {feature} with values: {test_values}")
                
                # Try each value in our smaller grid
                for value in test_values:
                    if value == current_value:
                        continue
                        
                    test_config = base_config.copy()
                    test_config[feature] = value
                    
                    # If we have a model, pre-screen with the model first
                    if hasattr(self, 'model') and self.model is not None:
                        predicted_score = self.predict_score(test_config)
                        if predicted_score < best_score:
                            print(f"    Skipping value {value}V (predicted score: {predicted_score:.2f})")
                            continue
                    
                    score, _ = self.evaluate_configuration(test_config)
                    
                    if score > best_score:
                        best_score = score
                        best_value = value
                        print(f"    New best value: {best_value}V (score: {best_score:.2f})")
                
                # Update base config with best value
                base_config[feature] = best_value

        return base_config
    
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
            popsize=20,
            maxiter=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            strategy='rand1bin',
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
    
    def optimize(self, full_optimization=True, initial_samples=1000):
        """Run multi-stage optimization for decoder"""
        print("Starting decoder optimization...")
            
        # Determine optimization approach based on existing data
        existing_data = len(self.configs_evaluated) > 0
        
        # If we have existing data and don't want full optimization
        if existing_data and not full_optimization:
            print(f"Building on {len(self.configs_evaluated)} existing evaluations.")
            
            # Train model on existing data
            self.train_surrogate_model()
            
            # Run evolution with existing model
            self.evolution_step(population_size=8, generations=3)
            
            # Update model
            self.train_surrogate_model()
            
            # Local search for fine-tuning
            final_config, final_score = self.local_search(iterations=10)
            
        else:
            # Full optimization process
            # Phase 1: Initial exploration (reduced samples if we have data)
            sample_count = initial_samples
            if existing_data:
                sample_count = max(10, initial_samples)  # Reduce sampling when building on data
                
            self.initial_sampling(n_samples=sample_count)
            
            # Phase 2: Train initial surrogate model
            self.train_surrogate_model()
            
            # Phase 3: Evolutionary optimization
            self.evolution_step(population_size=8, generations=3)
            
            # Phase 4: Retrain surrogate model  
            self.train_surrogate_model()
            
            # Phase 5: Adaptive grid search
            # With this line for a faster search:
            self.adaptive_grid_search(n_iterations=3, feature_count=5, grid_points=25)
            # Phase 6: Differential evolution
            self.differential_evolution_step()
            
            # Phase 7: Final local search
            final_config, final_score = self.local_search(iterations=10)
        
        print("\nOptimization complete!")
        print(f"Best score: {final_score:.2f}")
        
        # Save final data
        self.save_data()
        
        return final_config, final_score
    
    def test_final_configuration(self, config):
        """Test and print performance of the optimized decoder"""
        print("\nTesting final decoder configuration:")
        all_correct = True
        
        for test_config in TEST_CONFIGURATIONS:
            input_a, input_b = test_config
            
            # Determine which output should be active based on inputs
            expected_output_idx = (1 if input_b > LOW_THRESHOLD else 0) + 2 * (1 if input_a > LOW_THRESHOLD else 0)
            
            # Set up current test configuration
            current_config = config.copy()
            current_config[INPUT_PINS[0]] = input_a
            current_config[INPUT_PINS[1]] = input_b
            
            # Send configuration to hardware
            self.send_heater_values(current_config)
            time.sleep(0.25)
            
            # Measure all outputs
            outputs = self.measure_outputs()
            
            # Check if the correct output is active and others are inactive
            is_correct = (outputs[expected_output_idx] > HIGH_THRESHOLD and 
                         all(out < LOW_THRESHOLD for i, out in enumerate(outputs) if i != expected_output_idx))
            
            if not is_correct:
                all_correct = False
            
            # Determine status of each output (HIGH/LOW)
            output_states = ["HIGH" if out > HIGH_THRESHOLD else "LOW" for out in outputs]
            
            print(f"\nTest Case:")
            print(f"  Input Values (A, B): {input_a}V, {input_b}V")
            print(f"  Input Code: {int(input_a > HIGH_THRESHOLD)}{int(input_b > HIGH_THRESHOLD)}")
            print(f"  Expected Active Output: Channel {expected_output_idx+1}")
            print(f"  Measured Outputs:")
            for i, (out, state) in enumerate(zip(outputs, output_states)):
                print(f"    Channel {i+1}: {out:.4f}V ({state})")
            print(f"  Correct Decoding: {'Yes' if is_correct else 'No'}")
        
        if all_correct:
            print("\nSuccess! The decoder is working correctly for all input combinations.")
        else:
            print("\nThe decoder is not working perfectly for all input combinations.")
        
        return all_correct

    def list_saved_configurations(self):
        """List all the best configurations that have been saved"""
        if not os.path.isfile(BEST_CONFIG_PATH):
            print("No saved configurations found.")
            return []
        
        try:
            with open(BEST_CONFIG_PATH, 'r') as f:
                configs = json.load(f)
            
            print(f"\nFound {len(configs)} saved configurations:")
            for i, config in enumerate(configs):
                print(f"{i+1}. Time: {config['timestamp']}, Score: {config['score']:.2f}, "
                     f"Success: {config.get('success_count', '?')}")
            
            return configs
        except Exception as e:
            print(f"Error loading configurations: {e}")
            return []
    
    def load_specific_configuration(self, index):
        """Load a specific saved configuration by index"""
        configs = self.list_saved_configurations()
        
        if not configs:
            return None
        
        if index < 1 or index > len(configs):
            print(f"Invalid index. Please choose between 1 and {len(configs)}.")
            return None
        
        selected_config = configs[index-1]
        
        # Convert string keys back to integers
        config_dict = {}
        for k, v in selected_config['config'].items():
            config_dict[int(k)] = float(v)
        
        print(f"\nLoaded configuration from {selected_config['timestamp']}")
        print(f"Score: {selected_config['score']:.2f}")
        
        return config_dict
    
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
            if h not in INPUT_PINS:  # Don't override input pins
                complete_config[h] = 0.01
                
        # Format all values to 2 decimal places
        return {k: round(float(v), 2) for k, v in complete_config.items()}
    

def main():
    # Create decoder optimizer
    optimizer = DecoderOptimizer()

    try:
        # Run optimization
        best_config, best_score = optimizer.optimize()
        
        # Print final heater configuration
        print("\nFinal Decoder Heater Configuration:")
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