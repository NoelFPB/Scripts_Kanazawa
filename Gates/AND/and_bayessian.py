import serial
import time
import pyvisa
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from itertools import product

# Serial port configuration 
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Heater configuration
INPUT_HEATERS = [36, 37]  # Our AND gate inputs
FIXED_FIRST_LAYER = list(range(33, 40))
MODIFIABLE_HEATERS = [i for i in range(33) if i not in INPUT_HEATERS]

# Input states for AND gate testing
INPUT_COMBINATIONS = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]

# Voltage options for discretization
VOLTAGE_OPTIONS = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.9]

class ImprovedBayesianOptimizer:
    def __init__(self):
        # Initialize oscilloscope
        self.scope = self._init_scope()
        # Initialize serial connection
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(1)
        
        # Initialize Gaussian Process model with a more suitable kernel for this problem
        # Using a smaller length_scale to allow for more complex patterns
        # Setting length_scale_bounds to prevent convergence warnings
        kernel = ConstantKernel(1.0) * Matern(
            nu=1.5,  # Less smooth assumption
            length_scale=0.5,  # Smaller length scale
            length_scale_bounds=(1e-3, 10.0)  # Prevent hitting the lower bound
        ) + WhiteKernel(noise_level=0.1)
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=1e-2,  # Larger alpha to avoid numerical issues
            normalize_y=True, 
            n_restarts_optimizer=3  # Fewer restarts to speed up fitting
        )
        
        # Initialize observation history
        self.X_observed = []
        self.y_observed = []
        
        # Track best configuration seen so far
        self.best_config = None
        self.best_score = float('-inf')
        
    def _init_scope(self):
        """Initialize the oscilloscope and set up channels"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000

        # Setup only Channel 1 for AND gate output measurement
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
        """Measure the AND gate output from oscilloscope"""
        try:
            value = float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel1'))
            return round(value, 5)
        except Exception as e:
            print(f"Measurement error: {e}")
            return None
    
    def evaluate_configuration(self, config):
        """Evaluate a configuration across all input combinations"""
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
            
            # Enhanced scoring function for AND gate behavior
            if input_state == (4.9, 4.9):
                # Both inputs HIGH -> output should be HIGH
                if output > 3.0:  # Higher threshold for clearer distinction
                    # Quadratic reward for higher voltage - encourages stronger HIGH signal
                    total_score += 30 + (output * output)
                elif output > 2.0:
                    # Linear reward for acceptable HIGH signal
                    total_score += 20 + (output * 5)
                else:
                    # Severe penalty for incorrect HIGH signal
                    total_score -= 30
            else:
                # Any other input combination -> output should be LOW
                if output < 1:  # Stricter threshold for LOW
                    # Exponential reward for lower voltage
                    total_score += 20 + (10 * np.exp(-output))
                elif output < 1.5:
                    # Linear reward for acceptable LOW signal
                    total_score += 10 + (10 * (1.0 - output))
                else:
                    # Severe penalty for incorrect LOW signal - quadratic with output level
                    total_score -= 20 + (output * output)
        
        return total_score
    
    def config_to_vector(self, config):
        """Convert a configuration dictionary to a feature vector"""
        return np.array([[config[h] for h in MODIFIABLE_HEATERS]])
    
    def vector_to_config(self, vector):
        """Convert a feature vector to a configuration dictionary"""
        config = {h: vector[0][i] for i, h in enumerate(MODIFIABLE_HEATERS)}
        
        # Add fixed first layer values
        for h in FIXED_FIRST_LAYER:
            if h not in INPUT_HEATERS:
                config[h] = 0.01
                
        return config
    
    def expected_improvement(self, X, xi=0.1):  # Increased xi for more exploration
        """Calculate expected improvement acquisition function with exploration bonus"""
        # Get mean and standard deviation from GP model
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # Get current best
        if not self.y_observed:
            return np.zeros(X.shape[0])
        
        mu_sample = max(self.y_observed)
        
        # Calculate improvement
        with np.errstate(divide='warn'):
            imp = mu - mu_sample - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei
    
    def upper_confidence_bound(self, X, beta=2.0):
        """Upper Confidence Bound acquisition function (alternative to EI)"""
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + beta * sigma
    
    def generate_latin_hypercube_samples(self, n_samples):
        """Generate Latin Hypercube samples for better initial exploration"""
        # For each dimension, divide the range into n_samples equal bins
        n_dims = len(MODIFIABLE_HEATERS)
        
        # Generate samples
        result = np.zeros((n_samples, n_dims))
        for i in range(n_dims):
            # Generate a permutation of segment indices
            perm = np.random.permutation(n_samples)
            
            # For each sample, pick a random point within its assigned segment
            for j in range(n_samples):
                # Map to voltage option index (discretized Latin hypercube)
                bin_idx = int(perm[j] * len(VOLTAGE_OPTIONS) / n_samples)
                result[j, i] = VOLTAGE_OPTIONS[bin_idx]
                
        return result
    
    def generate_all_nearby_configurations(self, config, distance=1):
        """Generate all configurations that differ from the current one by at most 'distance' heaters"""
        result = []
        
        # Get baseline vector
        base_vector = self.config_to_vector(config)[0]
        
        # For efficiency, only try changing a subset of heaters at a time
        # (changing all 33 heaters would generate too many combinations)
        for heater_indices in self._get_heater_subsets(distance):
            # Try all voltage combinations for these heaters
            current_values = [base_vector[i] for i in heater_indices]
            for voltage_combo in product(VOLTAGE_OPTIONS, repeat=len(heater_indices)):
                if voltage_combo == tuple(current_values):
                    continue  # Skip the current configuration
                
                # Create new vector
                new_vector = base_vector.copy()
                for i, idx in enumerate(heater_indices):
                    new_vector[idx] = voltage_combo[i]
                
                result.append(new_vector.reshape(1, -1))
        
        if not result:  # Ensure we have at least some candidates
            return np.array([base_vector.reshape(1, -1)])
        
        return np.vstack(result)
    
    def _get_heater_subsets(self, max_size):
        """Get subsets of heater indices to modify"""
        import itertools
        
        # For efficiency, divide heaters into groups
        n_heaters = len(MODIFIABLE_HEATERS)
        group_size = 5  # Adjust as needed - smaller value = more focused search
        
        groups = [range(i, min(i + group_size, n_heaters)) for i in range(0, n_heaters, group_size)]
        
        # Generate combinations of 1, 2, ..., max_size heaters from each group
        result = []
        for group in groups:
            for size in range(1, min(max_size + 1, len(group) + 1)):
                result.extend(itertools.combinations(group, size))
        
        return result
    
    def sample_next_point(self, acquisition_func='ei'):
        """Sample the next point to evaluate based on acquisition function"""
        if len(self.X_observed) < 5:
            # Not enough data to fit a GP yet, return a Latin Hypercube sample
            samples = self.generate_latin_hypercube_samples(1)
            return samples[0].reshape(1, -1)
        
        # Generate candidates close to our current best configuration
        candidates = self.generate_all_nearby_configurations(self.best_config, distance=2)
        
        # Also include some random samples for exploration
        n_random = 200
        random_samples = np.random.choice(VOLTAGE_OPTIONS, size=(n_random, len(MODIFIABLE_HEATERS)))
        all_candidates = np.vstack([candidates, random_samples])
        
        # Calculate acquisition function values
        if acquisition_func == 'ei':
            acq_values = self.expected_improvement(all_candidates)
        else:  # UCB
            acq_values = self.upper_confidence_bound(all_candidates)
        
        # Return the point with highest acquisition value
        best_idx = np.argmax(acq_values)
        return all_candidates[best_idx].reshape(1, -1)
    
    def optimize(self, n_iterations=25, n_init=10, acquisition_func='ei'):
        """Run Bayesian optimization for AND gate configuration"""
        print("Starting Bayesian optimization for AND gate...")
        
        # Initial exploration phase
        print(f"Performing initial exploration with {n_init} configurations...")
        init_samples = self.generate_latin_hypercube_samples(n_init)
        
        for i, sample in enumerate(init_samples):
            # Convert to config dictionary
            config = self.vector_to_config(sample.reshape(1, -1))
            
            # Evaluate configuration
            score = self.evaluate_configuration(config)
            print(f"Initial configuration {i+1}/{n_init}: Score = {score}")
            
            # Track best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = config.copy()
                print(f"New best score: {self.best_score}")
            
            # Store observation
            self.X_observed.append(self.config_to_vector(config))
            self.y_observed.append(score)
        
        # Convert observations to numpy arrays and fit GP model
        X_np = np.vstack(self.X_observed)
        y_np = np.array(self.y_observed)
        self.gp.fit(X_np, y_np) # we are making a matrix here
        
        # Main optimization loop - adaptive acquisition function
        print("\nStarting main optimization phase...")
        for iteration in range(n_iterations):
            # Alternate between acquisition functions for better exploration/exploitation balance
            current_acq_func = 'ucb' if iteration % 3 == 0 else 'ei'
            
            # Get next point to evaluate
            next_point = self.sample_next_point(acquisition_func=current_acq_func)
            
            # Convert to config dictionary
            next_config = self.vector_to_config(next_point)
            
            # Evaluate configuration
            score = self.evaluate_configuration(next_config)
            print(f"Iteration {iteration+1}/{n_iterations}: Score = {score} (using {current_acq_func})")
            
            # Track best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = next_config.copy()
                print(f"New best score: {self.best_score}")
                
                # Local optimization around best configuration
                if score > self.best_score * 0.9:  # If we find a promising config
                    print("Performing local optimization around current best...")
                    local_best = self.local_search(next_config)
                    if local_best[0] > self.best_score:
                        self.best_score = local_best[0]
                        self.best_config = local_best[1].copy()
                        print(f"Local search improved score to: {self.best_score}")
            
            # Store observation
            self.X_observed.append(self.config_to_vector(next_config))
            self.y_observed.append(score)
            
            # Update GP model
            X_np = np.vstack(self.X_observed)
            y_np = np.array(self.y_observed)
            self.gp.fit(X_np, y_np)
        
        return self.best_config, self.best_score
    
    def local_search(self, config, iterations=5):
        """Perform a quick local search around a promising configuration"""
        best_config = config.copy()
        best_score = float('-inf')
        
        # Evaluate starting configuration
        score = self.evaluate_configuration(config)
        if score > best_score:
            best_score = score
            best_config = config.copy()
        
        # Try changing one heater at a time
        for _ in range(iterations):
            improved = False
            
            # Shuffle heaters to avoid bias
            heaters = list(MODIFIABLE_HEATERS)
            np.random.shuffle(heaters)
            
            for heater in heaters:
                current_value = best_config[heater]
                
                # Try neighbor voltage values
                neighbor_values = [v for v in VOLTAGE_OPTIONS if abs(VOLTAGE_OPTIONS.index(v) - VOLTAGE_OPTIONS.index(current_value)) <= 2]
                
                for new_value in neighbor_values:
                    if new_value == current_value:
                        continue
                        
                    test_config = best_config.copy()
                    test_config[heater] = new_value
                    
                    score = self.evaluate_configuration(test_config)
                    
                    if score > best_score:
                        best_score = score
                        best_config = test_config.copy()
                        improved = True
                        print(f"  Local improvement: Heater {heater} = {new_value}V, Score = {best_score}")
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best_score, best_config
    
    def test_final_configuration(self, config):
        """Test and print performance of the optimized configuration"""
        print("\nTesting final AND gate configuration:")
        all_correct = True
        
        for input_state in INPUT_COMBINATIONS:
            current_config = config.copy()
            current_config[INPUT_HEATERS[0]] = input_state[0]
            current_config[INPUT_HEATERS[1]] = input_state[1]
            
            self.send_heater_values(current_config)
            time.sleep(0.25)
            output_value = self.measure_output()
            
            is_high = output_value > 2.0
            expected_high = input_state == (4.9, 4.9)
            is_correct = is_high == expected_high
            
            if not is_correct:
                all_correct = False
            
            print(f"\nInputs (A, B): {input_state}")
            print(f"AND Output: {output_value:.4f}V")
            print(f"Output is: {'HIGH' if is_high else 'LOW'}")
            print(f"Expected: {'HIGH' if expected_high else 'LOW'}")
            print(f"Correct: {'Yes' if is_correct else 'No'}")
        
        if all_correct:
            print("\nSuccess! The AND gate is working correctly for all input combinations.")
        else:
            print("\nThe AND gate is not working perfectly for all input combinations.")
        
        return all_correct
    
    def cleanup(self):
        """Close connections"""
        self.serial.close()
        self.scope.close()
        print("Connections closed.")


def main():
    optimizer = ImprovedBayesianOptimizer()
    
    try:
        # Run Bayesian optimization
        best_config, best_score = optimizer.optimize(
            n_iterations=25,  # More iterations
            n_init=10,        # More initial points
            acquisition_func='ei'
        )
        
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