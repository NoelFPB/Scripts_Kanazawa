import numpy as np
import time
import csv
import pyvisa as visa
import serial
import pandas as pd
from sklearn.model_selection import train_test_split
import random

class ConfigurationManager:
    """Manages configuration generation and validation"""
    def __init__(self):
        self.modifiable_heaters = sorted([i for i in range(35)], reverse=True)
        self.fixed_inputs = list(range(36, 40))
        self.voltage_options = [0.1, 1.0, 2.0, 3.0, 4.0, 4.9]  # Expanded voltage options

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
        self.channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
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
    def __init__(self, csv_path, output_path):
        self.csv_path = csv_path
        self.output_path = output_path
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
        """Calculate Mean Squared Error for outputs"""
        target = self.target[number]
        real = np.array([real1, real2, real3])
        ratio = real / np.sum(real)
        mse = (np.sum((target - ratio)**2) / 3)
        return round(mse, 5)

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
        time.sleep(0.001)

    def train(self, config_manager, data_processor, oscilloscope, population_size=10, generations=5, mutation_rate=0.1):
        """Training phase using genetic algorithm to find optimal configurations"""
        print("\nStarting Training Phase...")
        
        # Use stored training indices
        train_data = data_processor.train_indices
        
        # Initialize population
        population = [config_manager.generate_random_config() for _ in range(population_size)]
        
        for generation in range(generations):
            print(f"\nGeneration {generation + 1}/{generations}:")
            fitness = []
            
            for idx, config in enumerate(population):
                generation_mse = []
                print(f"\nTesting configuration {idx + 1}/{population_size}")
                
                    # Test configuration against training samples
                for sample_id in train_data:
                    # Combine input and heater configurations
                    iris_data = data_processor.df.iloc[sample_id]
                    input_config = config_manager.generate_input_config(iris_data)
                    
                    # Merge both configurations
                    combined_config = {**config, **input_config}  # This combines both dictionaries
                    
                    # Send combined configuration once
                    self.send_heater_values(combined_config)
                    time.sleep(2)
                        # Measure outputs
                    outputs = oscilloscope.measure_outputs()
                    if outputs[0] is not None:
                        mse = data_processor.calculate_mse(*outputs, sample_id)
                        generation_mse.append(mse)
                        print(f"Sample {sample_id} MSE: {mse:.5f}")
                
                avg_mse = np.mean(generation_mse) if generation_mse else float('inf')
                fitness.append((config, avg_mse))
                print(f"Configuration average MSE: {avg_mse:.5f}")
            
            # Update best configurations
            fitness.sort(key=lambda x: x[1])
            self.best_configurations = fitness[:5]
            print(f"\nBest MSE this generation: {fitness[0][1]:.5f}")
            
            # Generate next population
            next_population = []
            while len(next_population) < population_size:
                # Tournament selection
                tournament = random.sample(fitness, 3)
                parent1 = min(tournament, key=lambda x: x[1])[0]
                tournament = random.sample(fitness, 3)
                parent2 = min(tournament, key=lambda x: x[1])[0]
                
                child = config_manager.crossover(parent1, parent2)
                child = config_manager.mutate(child, mutation_rate)
                next_population.append(child)
            
            population = next_population
        
        print("\nTraining Complete!")
        print("Best configurations found:")
        for i, (config, mse) in enumerate(self.best_configurations):
            print(f"Rank {i+1} - MSE: {mse:.5f}")

    def test(self, data_processor, oscilloscope, config_manager):
        """Testing phase using the best configuration found during training"""
        print("\nStarting Testing Phase...")
        
        # Use stored test indices
        test_data = data_processor.test_indices
        
        if not self.best_configurations:
            print("No trained configurations found. Please run training first.")
            return
        
        # Use the best configuration from training
        best_config = self.best_configurations[0][0]
        print(f"\nTesting best configuration:")
        print(f"Configuration: {best_config}")
        
        test_mse = []
        for sample_id in test_data:
            # Set input configuration
            iris_data = data_processor.df.iloc[sample_id]
            input_config = config_manager.generate_input_config(iris_data)
            self.send_heater_values(input_config)
            time.sleep(2)
            
            # Apply best configuration
            self.send_heater_values(best_config)
            time.sleep(2)
            
            # Measure and calculate MSE
            outputs = oscilloscope.measure_outputs()
            if outputs[0] is not None:
                mse = data_processor.calculate_mse(*outputs, sample_id)
                test_mse.append(mse)
                print(f"Test sample {sample_id} MSE: {mse:.5f}")
        
        avg_test_mse = np.mean(test_mse)
        print(f"\nOverall Test MSE: {avg_test_mse:.5f}")

def main():
    oscilloscope = OscilloscopeController()
    config_manager = ConfigurationManager()
    data_processor = DataProcessor(
        csv_path='C:\\Users\\noelp\\Documents\\Kanazawa\\Scripts_Kanazawa\\MSE\\iris2.csv',
        output_path='C:\\Users\\noelp\\Documents\\Kanazawa\\Scripts_Kanazawa\\MSE\\large_scale.csv'
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
    
    # Training phase
    serial_controller.train(
        config_manager=config_manager,
        data_processor=data_processor,
        oscilloscope=oscilloscope,
        population_size=10,
        generations=5,
        mutation_rate=0.1
    )
    
    # Testing phase
    serial_controller.test(
        data_processor=data_processor,
        oscilloscope=oscilloscope,
        config_manager=config_manager
    )

if __name__ == "__main__":
    main()