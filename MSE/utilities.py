import numpy as np
import time
import pyvisa as visa
import serial
import pandas as pd
from sklearn.model_selection import train_test_split
import random


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

# utils.py
# ... [keep previous imports and other classes the same] ...

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
        self.train_indices, self.test_indices = train_test_split(
            self.df['Id'], 
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
        """Calculate error based on maximum output matching target"""
        target = self.target[number]
        outputs = np.array([real1, real2, real3])
        
        # Get index of maximum output and target
        max_output_index = np.argmax(outputs)
        target_index = np.argmax(target)
        
        # Calculate base score based on correct maximum
        if max_output_index == target_index:
            # If the maximum is in the correct position, calculate separation score
            sorted_outputs = np.sort(outputs)
            max_val = sorted_outputs[-1]
            second_max = sorted_outputs[-2]
            
            # Calculate separation ratio (how much larger is the maximum compared to others)
            separation = (max_val - second_max) / max_val if max_val != 0 else 0
            
            # Return error (0 is perfect, 1 is worst)
            # Scale separation to be between 0 and 0.5
            # This way, a correct maximum with poor separation will still be better
            # than an incorrect maximum
            return 0.5 * (1 - separation)
        else:
            # If maximum is in wrong position, return worst score
            return 1.0

class ConfigurationManager:
    """Manages configuration generation and validation"""
    def __init__(self):
        self.modifiable_heaters = sorted([i for i in range(35)], reverse=True)
        self.fixed_inputs = list(range(36, 40))
        self.voltage_options = [0.1, 2.5, 4.9]

    def generate_random_config(self):
        """Generate a random initial configuration"""
        return {h: random.choice(self.voltage_options) for h in self.modifiable_heaters}

    def generate_input_config(self, iris_data):
        """Create configuration for fixed inputs based on iris data"""
        return {
            36: iris_data['SeL'],
            37: iris_data['SeW'],
            38: iris_data['PeL'],
            39: iris_data['PeW']
        }

class SerialController:
    def __init__(self, port='COM4', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.best_configuration = None
        self.best_mse = float('inf')

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
        time.sleep(2)

    def evaluate_configuration(self, config, config_manager, data_processor, oscilloscope, train_data):
        """Evaluate a single configuration across all training samples"""
        mse_scores = []
        
        for sample_id in train_data:
            iris_data = data_processor.df.iloc[sample_id]
            input_config = config_manager.generate_input_config(iris_data)
            combined_config = {**config, **input_config}
            
            self.send_heater_values(combined_config)
            outputs = oscilloscope.measure_outputs()
            
            if outputs[0] is not None:
                mse = data_processor.calculate_mse(*outputs, sample_id)
                mse_scores.append(mse)
        
        return np.mean(mse_scores) if mse_scores else float('inf')

    def hill_climbing(self, config_manager, data_processor, oscilloscope, iterations=20):
        """Hill climbing optimization for iris classification"""
        print("\nStarting Hill Climbing Optimization...")
        
        # Initialize with random configuration
        current_config = config_manager.generate_random_config()
        current_mse = self.evaluate_configuration(
            current_config, 
            config_manager, 
            data_processor, 
            oscilloscope,
            data_processor.train_indices
        )
        
        self.best_configuration = current_config.copy()
        self.best_mse = current_mse
         
        print(f"Initial MSE: {current_mse:.5f}")
        
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")
            improved = False
            
            # Try to improve each heater
            for heater in config_manager.modifiable_heaters:
                current_value = current_config[heater]
                
                # Try different voltage values
                for new_value in config_manager.voltage_options:
                    if new_value != current_value:
                        test_config = current_config.copy()
                        test_config[heater] = new_value
                        
                        mse = self.evaluate_configuration(
                            test_config,
                            config_manager,
                            data_processor,
                            oscilloscope,
                            data_processor.train_indices
                        )
                        
                        if mse < current_mse:
                            current_config = test_config.copy()
                            current_mse = mse
                            improved = True
                            
                            if mse < self.best_mse:
                                self.best_configuration = test_config.copy()
                                self.best_mse = mse
                                print(f"New best MSE: {self.best_mse:.5f}")
                            break
                
                if improved:
                    break
            
            if not improved:
                print("No improvement found in this iteration")
                break
        
        print("\nHill Climbing Complete!")
        print(f"Best MSE found: {self.best_mse:.5f}")
        return self.best_configuration, self.best_mse

    def test(self, data_processor, oscilloscope, config_manager):
        """Testing phase using the best configuration found"""
        print("\nStarting Testing Phase...")
        
        if not self.best_configuration:
            print("No optimized configuration found. Please run hill climbing first.")
            return
        
        test_mse = []
        for sample_id in data_processor.test_indices:
            iris_data = data_processor.df.iloc[sample_id]
            input_config = config_manager.generate_input_config(iris_data)
            combined_config = {**self.best_configuration, **input_config}
            
            self.send_heater_values(combined_config)
            outputs = oscilloscope.measure_outputs()
            
            if outputs[0] is not None:
                mse = data_processor.calculate_mse(*outputs, sample_id)
                test_mse.append(mse)
                print(f"Test sample {sample_id} MSE: {mse:.5f}")
        
        avg_test_mse = np.mean(test_mse)
        print(f"\nOverall Test MSE: {avg_test_mse:.5f}")
