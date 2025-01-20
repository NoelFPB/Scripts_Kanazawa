import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import time
import serial
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import random
import pyvisa as visa

class ConfigurationManager:
    """Manages configuration generation and validation"""
    def __init__(self):
        self.modifiable_heaters = sorted([i for i in range(35)], reverse=True)
        self.fixed_inputs = list(range(36, 40))
        self.voltage_options = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.9]
    def generate_random_config(self):
        """Generate a random configuration with diverse voltage settings"""
        config = {
            h: random.choice(self.voltage_options)
            for h in self.modifiable_heaters
        }
        return config

    def generate_input_config(self, iris_data):
        """Create configuration for fixed inputs based on iris data"""
        return {
            36: iris_data['SeL'],
            37: iris_data['SeW'],
            38: iris_data['PeL'],
            39: iris_data['PeW']
        }

    def crossover(self, parent1, parent2):
        """Perform crossover between two parent configurations"""
        child = {}
        for heater in parent1:
            child[heater] = parent1[heater] if random.random() < 0.5 else parent2[heater]
        return child

    def mutate(self, config, mutation_rate=0.1):
        """Mutate a configuration with a given mutation rate"""
        for heater in config:
            if random.random() < mutation_rate:
                config[heater] = random.choice(self.voltage_options)
        return config

class OscilloscopeController:
    def __init__(self):
        self.channels = ['CHANnel1', 'CHANnel2', 'CHANnel3']
        self.scope = None

    def connect_oscilloscope(self):
        try:
            rm = visa.ResourceManager()
            resources = rm.list_resources()
            if not resources:
                raise Exception("No VISA resources found")
            self.scope = rm.open_resource(resources[0])
            self.initialize_channels()
            return True
        except Exception as e:
            print(f"Failed to connect to oscilloscope: {e}")
            return False

    def initialize_channels(self):
        if not self.scope:
            return
        for channel in self.channels:
            self.scope.write(f':{channel}:DISPlay ON')
            self.scope.write(f':{channel}:SCALe 2')
            self.scope.write(f':{channel}:OFFSet -6')

    def measure_outputs(self):
        try:
            outputs = []
            for channel in range(1, 4):
                value = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                outputs.append(round(value, 5))
            return outputs
        except Exception as e:
            print(f"Measurement error: {e}")
            return [None] * 3

class DataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.target = []
        self.train_indices = None
        self.test_indices = None

    def load_data(self):
        """Load data and create train/test split"""
        self.df = pd.read_csv(self.csv_path)
        self._create_target_encoding()
        # Create and store the train/test split
        self.train_indices, self.test_indices = train_test_split(
            self.df['Id'].apply(lambda x: x - 1), 
            test_size=0.3, 
            random_state=42
        )

    def _create_target_encoding(self):
        """Create one-hot encoding for iris species"""
        for species in self.df['Species']:
            if species == "Iris-setosa":
                self.target.append([1, 0, 0])
            elif species == "Iris-versicolor":
                self.target.append([0, 1, 0])
            elif species == "Iris-virginica":
                self.target.append([0, 0, 1])

    def calculate_mse(self, real1, real2, real3, number):
        """Calculate error using both classification and distribution matching"""
        target = self.target[number]
        real = np.array([real1, real2, real3])
        
        # Classification error (0 or 1)
        predicted_class = np.argmax(real)
        true_class = np.argmax(target)
        class_error = 0 if predicted_class == true_class else 1
        
        # Distribution error
        real_normalized = real / np.sum(real)
        dist_error = np.mean((np.array(target) - real_normalized) ** 2)
        
        # Combine both (weighting can be adjusted)
        return 0.7 * class_error + 0.3 * dist_error
 
    # def calculate_mse(self, real1, real2, real3, number):
    #     """Calculate error based on whether the highest output matches the target class"""
    #     target = self.target[number]
    #     real = np.array([real1, real2, real3])
        
    #     # Find the index of the maximum value in both arrays
    #     predicted_class = np.argmax(real)
    #     true_class = np.argmax(target)
        
    #     # Return 0 if the prediction is correct, 1 if incorrect
    #     error = 0 if predicted_class == true_class else 1
        
    #     return error
    # def calculate_mse(self, real1, real2, real3, number):
    #     """Calculate error using continuous values"""
    #     target = self.target[number]
    #     real = np.array([real1, real2, real3])
        
    #     # Normalize the outputs
    #     real_normalized = real / np.sum(real)
        
    #     # Calculate mean squared error between actual and target
    #     mse = np.mean((np.array(target) - real_normalized) ** 2)
    #     return mse
    
class BayesianOptimizer:
    """Manages Bayesian optimization for heater configurations"""
    def __init__(self, voltage_options, n_heaters):
        self.voltage_options = np.array(voltage_options)
        self.n_heaters = n_heaters
        
        # Define GP kernel
        # Using Matern kernel which is more robust to non-smooth functions
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42
        )
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
        
    def _config_to_array(self, config):
        """Convert configuration dictionary to array"""
        return np.array([config[i] for i in range(self.n_heaters)])
    
    def _array_to_config(self, array):
        """Convert array to configuration dictionary"""
        return {i: float(v) for i, v in enumerate(array)}
    
    def _acquisition_function(self, X):
        """Upper Confidence Bound acquisition function"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get mean and std from GP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self.gp.predict(X, return_std=True)
            
        # UCB with dynamic exploration term
        kappa = np.sqrt(np.log(len(self.X_observed) + 1))
        return mean - kappa * std  # Negative because we're minimizing
    
    def suggest_configuration(self, n_random=1000):
        """Suggest next configuration to try"""
        if len(self.X_observed) < 5:
            # Initial random exploration
            config_array = np.random.choice(
                self.voltage_options, 
                size=self.n_heaters
            )
        else:
            # Fit GP to observed data
            X_observed = np.array(self.X_observed)
            y_observed = np.array(self.y_observed)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X_observed, y_observed)
            
            # Generate random candidates
            candidates = np.random.choice(
                self.voltage_options,
                size=(n_random, self.n_heaters)
            )
            
            # Select best candidate according to acquisition function
            acq_values = self._acquisition_function(candidates)
            best_idx = np.argmin(acq_values)
            config_array = candidates[best_idx]
        
        return self._array_to_config(config_array)
    
    def update(self, config, fitness):
        """Update the optimizer with new observation"""
        config_array = self._config_to_array(config)
        self.X_observed.append(config_array)
        self.y_observed.append(fitness)

class SerialController:
    def __init__(self, port='COM4', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.best_configurations = []

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate)
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Failed to connect to serial port: {e}")
            return False

    def send_heater_values(self, config):
        """Send heater configuration through serial port"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.ser.write(voltage_message.encode())
        self.ser.flush()
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def train_with_bayesian(self, config_manager, data_processor, oscilloscope, 
                           n_iterations=50, stabilization_time=2):
        """Training phase using Bayesian optimization"""
        print("\nStarting Bayesian Optimization Training...")
        
        # Initialize Bayesian optimizer
        optimizer = BayesianOptimizer(
            voltage_options=config_manager.voltage_options,
            n_heaters=len(config_manager.modifiable_heaters)
        )
        
        best_config = None
        best_fitness = float('inf')
        
        for iteration in range(n_iterations):
            print(f"\nIteration {iteration + 1}/{n_iterations}:")
            
            # Get next configuration to try
            config = optimizer.suggest_configuration()
            print("Testing configuration:", config)
            
            # Send configuration and wait for stabilization
            self.send_heater_values(config)
            time.sleep(stabilization_time)
            
            # Test configuration against training samples
            iteration_errors = []
            for sample_id in data_processor.train_indices:
                # Set input configuration
                iris_data = data_processor.df.iloc[sample_id]
                input_config = config_manager.generate_input_config(iris_data)
                self.send_heater_values(input_config)
                time.sleep(0.3)
                
                # Measure outputs
                outputs = oscilloscope.measure_outputs()
                if outputs[0] is not None:
                    error = data_processor.calculate_mse(*outputs, sample_id)
                    iteration_errors.append(error)
                    print(f"Sample {sample_id} Error: {error:.5f}")
            
            # Calculate average error for this configuration
            avg_error = np.mean(iteration_errors) if iteration_errors else float('inf')
            print(f"Configuration average error: {avg_error:.5f}")
            
            # Update optimizer with results
            optimizer.update(config, avg_error)
            
            # Update best configuration if better
            if avg_error < best_fitness:
                best_fitness = avg_error
                best_config = config
                print(f"New best configuration found! Error: {best_fitness:.5f}")
            
        # Store best configuration
        self.best_configurations = [(best_config, best_fitness)]
        print("\nTraining Complete!")
        print(f"Best configuration found - Error: {best_fitness:.5f}")
        print("Configuration:", best_config)

    def test(self, data_processor, oscilloscope, config_manager):
        """Testing phase using the best configuration found during training"""
        # Existing test method remains the same
        pass

def main():
    oscilloscope = OscilloscopeController()
    config_manager = ConfigurationManager()
    data_processor = DataProcessor(
        csv_path='C:\\Users\\noelp\\Documents\\Kanazawa\\Scripts_Kanazawa\\MSE\\Datasets\\iris_normalized.csv'
    )
    serial_controller = SerialController()

    # Setup connections
    if not oscilloscope.connect_oscilloscope():
        print("Failed to connect to oscilloscope")
        return
    if not serial_controller.connect():
        print("Failed to connect to serial port")
        return

    # Load data and create train/test split
    data_processor.load_data()
    
    # Training phase with Bayesian optimization
    serial_controller.train_with_bayesian(
        config_manager=config_manager,
        data_processor=data_processor,
        oscilloscope=oscilloscope,
        n_iterations=50,  # Adjust based on time constraints
        stabilization_time=2  # Adjust based on your system
    )
    
    # Testing phase
    serial_controller.test(
        data_processor=data_processor,
        oscilloscope=oscilloscope,
        config_manager=config_manager
    )

if __name__ == "__main__":
    main()