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

# === GATE CONFIGURATION - MODIFY THIS SECTION ===
# Logic gate type to optimize (AND, OR, NAND, NOR, XOR, XNOR, or CUSTOM)
GATE_TYPE = "OR"

# Input heaters for logic gate
INPUT_HEATERS = [36, 37]  # Our gate inputs

# For CUSTOM gate type, define truth table as dict of tuples (input_a, input_b) -> expected_output
# For standard gates, the truth table will be generated automatically
# Use True for HIGH (>2.0V), False for LOW (<2.0V)
# Example:
# CUSTOM_TRUTH_TABLE = {
#     (0.1, 0.1): False,  # 0 OP 0 = 0 (LOW)
#     (0.1, 4.9): True,   # 0 OP 1 = 1 (HIGH)
#     (4.9, 0.1): True,   # 1 OP 0 = 1 (HIGH)
#     (4.9, 4.9): True,   # 1 OP 1 = 1 (HIGH)
# }
CUSTOM_TRUTH_TABLE = {}

# Input voltage values representing LOW and HIGH states
LOW_VOLTAGE = 0.1    # Voltage representing logical LOW
HIGH_VOLTAGE = 4.9   # Voltage representing logical HIGH

# Output voltage thresholds
LOW_THRESHOLD = 1.5  # Outputs below this are considered LOW
OPTIMAL_LOW = 1

HIGH_THRESHOLD = 4.5 # Outputs above this start to be considered HIGH
OPTIMAL_HIGH = 4.9   # Optimal HIGH output


# Heater configuration
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in INPUT_HEATERS]

# Input states for gate testing will be generated based on selected gate type
INPUT_COMBINATIONS = [(LOW_VOLTAGE, LOW_VOLTAGE),
                     (LOW_VOLTAGE, HIGH_VOLTAGE),
                     (HIGH_VOLTAGE, LOW_VOLTAGE),
                     (HIGH_VOLTAGE, HIGH_VOLTAGE)]

# Voltage options for discretization
def create_voltage_options(start=0, end=4.9, step=0.1):
    options = []
    current = start
    options.append(0.1)
    
    while current <= end:
        options.append(round(current, 2))  # Round to 1 decimal place to avoid floating point issues
        current += step
        
    
    # Add the final value (4.9) if it's not already included
    if end not in options:
        options.append(end)
    if 0 in options:
        options.remove(0)
        
    return options

# Example usage with default step size of 0.5V
VOLTAGE_OPTIONS = create_voltage_options()
VOLTAGE_OPTIONS_GRID = create_voltage_options(start = 0, end = 4.9, step = 0.5)
print(VOLTAGE_OPTIONS)

# Generate truth table based on gate type
def generate_truth_table(gate_type):
    """Generate truth table based on selected gate type"""
    if gate_type == "CUSTOM" and CUSTOM_TRUTH_TABLE:
        return CUSTOM_TRUTH_TABLE
    
    truth_table = {}
    inputs = [(LOW_VOLTAGE, LOW_VOLTAGE), 
              (LOW_VOLTAGE, HIGH_VOLTAGE), 
              (HIGH_VOLTAGE, LOW_VOLTAGE), 
              (HIGH_VOLTAGE, HIGH_VOLTAGE)]
    
    if gate_type == "AND":
        outputs = [False, False, False, True]
    elif gate_type == "OR":
        outputs = [False, True, True, True]
    elif gate_type == "NAND":
        outputs = [True, True, True, False]
    elif gate_type == "NOR":
        outputs = [True, False, False, False]
    elif gate_type == "XOR":
        outputs = [False, True, True, False]
    elif gate_type == "XNOR":
        outputs = [True, False, False, True]
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")
    
    for i, input_pair in enumerate(inputs):
        truth_table[input_pair] = outputs[i]
    
    return truth_table

class LogicGateOptimizer:
    """
    Optimize a physical logic gate using ensemble learning and evolutionary strategies.
    """
    def __init__(self, gate_type=GATE_TYPE):
        # Initialize gate type and truth table
        self.gate_type = gate_type
        self.truth_table = generate_truth_table(gate_type)
        
        print(f"Optimizing {gate_type} gate with truth table:")
        for inputs, output in self.truth_table.items():
            print(f"  {inputs} -> {'HIGH' if output else 'LOW'}")
        
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
        """Initialize oscilloscope for logic gate output measurement"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000

        # Setup Channel 1 for logic gate output measurement
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
        """Measure the logic gate output voltage from oscilloscope"""
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
        Evaluate a configuration with emphasis on output state consistency.
        """
        # Check if already evaluated
        config_key = tuple(config.get(h, 0.1) for h in sorted(config.keys()))
        for i, past_config in enumerate(self.configs_evaluated):
            past_key = tuple(past_config.get(h, 0.1) for h in sorted(past_config.keys()))
            if config_key == past_key:
                return self.scores_evaluated[i], None
        
        # Component weights (sum to 1.0)
        WEIGHTS = {
            'high_state': 0.2,      # HIGH output performance
            'low_state': 0.2,       # LOW output performance
            'high_consistency': 0.20, # Consistency of HIGH outputs
            'low_consistency': 0.10,  # Consistency of LOW outputs
            'separation': 0.30       # Separation between HIGH/LOW
        }
        
        # Initialize scores
        component_scores = {k: 0.0 for k in WEIGHTS}
        results = []
        
        # Collect all outputs for consistency calculation
        high_outputs = []
        low_outputs = []
        
        # First pass - collect all outputs
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            expected_high = self.truth_table[input_state]
            
            self.send_heater_values(current_config)
            time.sleep(0.20)
            
            output = self.measure_output()
            if output is None:
                return -1000, []
            
            # Collect outputs based on expected state
            if expected_high:
                high_outputs.append(output)
            else:
                low_outputs.append(output)
                
            # Store results for scoring
            actual_high = output > LOW_THRESHOLD
            results.append({
                'inputs': input_state,
                'output': output,
                'expected': 'HIGH' if expected_high else 'LOW',
                'actual': 'HIGH' if actual_high else 'LOW',
                'correct': expected_high == actual_high
            })
        
        # If we have missing state combinations, penalize
        if not high_outputs or not low_outputs:
            return -500, results
        
        # Second pass - calculate individual state scores
        for result in results:
            output = result['output']
            expected_high = result['expected'] == 'HIGH'
            
            # Score for HIGH states
            if expected_high:
                if output >= OPTIMAL_HIGH:
                    high_score = 1.0
                elif output >= HIGH_THRESHOLD:
                    high_score = 0.5 + 0.5 * ((output - HIGH_THRESHOLD) / (OPTIMAL_HIGH - HIGH_THRESHOLD))
                else:
                    high_score = max(0.0, 0.5 * (output / HIGH_THRESHOLD))
                
                component_scores['high_state'] += high_score / len(high_outputs)
            
            # Score for LOW states
            else:
                if output <= OPTIMAL_LOW:
                    low_score = 1.0
                elif output <= LOW_THRESHOLD:
                    low_score = 0.5 + 0.5 * ((LOW_THRESHOLD - output) / (LOW_THRESHOLD - OPTIMAL_LOW))
                else:
                    low_score = max(0.0, 0.5 * (1 - min(1, (output - LOW_THRESHOLD) / 2)))
                
                component_scores['low_state'] += low_score / len(low_outputs)
        
        # Calculate consistency scores
        # For HIGH states
        if len(high_outputs) > 1:
            avg_high = sum(high_outputs) / len(high_outputs)
            high_variance = sum((o - avg_high)**2 for o in high_outputs) / len(high_outputs)
            # Convert variance to consistency score (lower variance = higher consistency)
            # Scale to reasonable voltage variance (0.25V² is considered good consistency)
            component_scores['high_consistency'] = 1.0 / (1.0 + (high_variance / 0.25))
        elif len(high_outputs) == 1:
            # Can't measure consistency with just one output
            component_scores['high_consistency'] = 0.5
        
        # For LOW states
        if len(low_outputs) > 1:
            avg_low = sum(low_outputs) / len(low_outputs)
            low_variance = sum((o - avg_low)**2 for o in low_outputs) / len(low_outputs)
            # Convert variance to consistency score
            component_scores['low_consistency'] = 1.0 / (1.0 + (low_variance / 0.25))
        elif len(low_outputs) == 1:
            component_scores['low_consistency'] = 0.5
        
        # Calculate separation between averages
        avg_high = sum(high_outputs) / len(high_outputs) if high_outputs else 0
        avg_low = sum(low_outputs) / len(low_outputs) if low_outputs else 0
        separation = avg_high - avg_low
        
        # Score separation (normalized to ideal separation)
        ideal_separation = OPTIMAL_HIGH - OPTIMAL_LOW
        component_scores['separation'] = min(1.0, separation / ideal_separation) if separation > 0 else 0.0
        
        # Calculate final score (0-100)
        final_score = sum(component_scores[k] * WEIGHTS[k] * 100 for k in WEIGHTS)
        
        # Add success bonus for completely correct configurations
        success_count = sum(1 for r in results if r['correct'])
        if success_count == len(INPUT_COMBINATIONS):
            final_score += 20  # Bonus for perfect behavior
        
        # Add to history
        self.configs_evaluated.append(config.copy())
        self.scores_evaluated.append(final_score)
        
        # Update best if improved
        if final_score > self.best_score:
            self.best_score = final_score
            self.best_config = config.copy()
            print(f"\nNew best score: {final_score:.2f}")
            print(f"Components: HIGH={component_scores['high_state']:.2f}, " 
                f"LOW={component_scores['low_state']:.2f}, "
                f"HIGH_CONS={component_scores['high_consistency']:.2f}, "
                f"LOW_CONS={component_scores['low_consistency']:.2f}, "
                f"SEP={component_scores['separation']:.2f}")
            print(f"Success count: {success_count}/{len(INPUT_COMBINATIONS)}")
            
            if high_outputs:
                print(f"HIGH outputs: avg={sum(high_outputs)/len(high_outputs):.2f}V, " 
                    f"min={min(high_outputs):.2f}V, max={max(high_outputs):.2f}V")
            if low_outputs:
                print(f"LOW outputs: avg={sum(low_outputs)/len(low_outputs):.2f}V, "
                    f"min={min(low_outputs):.2f}V, max={max(low_outputs):.2f}V")
        
        return final_score, results

    def initial_sampling(self, n_samples=10):
        """Initial sampling using Latin Hypercube Sampling (LHS) for better space coverage"""
        from scipy.stats import qmc  # More standard library for experimental design
        
        print(f"Performing Latin Hypercube initial sampling with {n_samples} configurations...")
        
        # Zero configuration (baseline)
        zero_config = {h: 0.1 for h in MODIFIABLE_HEATERS}
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
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
                if h not in INPUT_HEATERS:
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
                if h not in INPUT_HEATERS:
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
            return -self.predict_score(config)  # N     egative because we're minimizing
        
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
        """Run multi-stage optimization for logic gate"""
        print(f"Starting {self.gate_type} gate optimization...")
        
        # Phase 1: Initial exploration
        self.initial_sampling(n_samples=20)
        
        # Phase 2: Train initial surrogate model
        self.train_surrogate_model()
        
        # Phase 3: Evolutionary optimization
        self.evolution_step(population_size=16  , generations=3)
        
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
        print(f"\nTesting final {self.gate_type} gate configuration:")
        all_correct = True
        
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            output_value = self.measure_output()
                        
            # Check if output is in the right range based on expected output
            expected_high = self.truth_table[input_state]
            
            if expected_high:
                is_correct = output_value > HIGH_THRESHOLD
                output_state = "HIGH" if output_value > HIGH_THRESHOLD else "LOW"
            else:
                is_correct = output_value < LOW_THRESHOLD
                output_state = "LOW" if output_value < LOW_THRESHOLD else "HIGH"

            if not is_correct:
                all_correct = False

            print(f"\nInputs (A, B): {input_state}")
            print(f"{self.gate_type} Output: {output_value:.4f}V")
            print(f"Output is: {output_state}")
            print(f"Expected: {'HIGH' if expected_high else 'LOW'}")
            print(f"Correct: {'Yes' if is_correct else 'No'}")
                    
        if all_correct:
            print(f"\nSuccess! The {self.gate_type} gate is working correctly for all input combinations.")
        else:
            print(f"\nThe {self.gate_type} gate is not working perfectly for all input combinations.")
        
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
    # Select gate type at runtime - modify this line to change gate type
    gate_type = GATE_TYPE  # Use the value from configuration section
    
    # Create optimizer for the selected gate type
    optimizer = LogicGateOptimizer(gate_type)
    
    try:
        # Run optimization
        best_config, best_score = optimizer.optimize()
        
        # Print final heater configuration
        print(f"\nFinal {gate_type} Gate Heater Configuration:")
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