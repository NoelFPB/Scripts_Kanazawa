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

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self._create_target_encoding()

    def _create_target_encoding(self):
        for species in self.df['Species']:
            if species == "Iris-setosa":
                self.target.append([1, 0, 0])
            elif species == "Iris-versicolor":
                self.target.append([0, 1, 0])
            elif species == "Iris-virginica":
                self.target.append([0, 0, 1])

    def calculate_mse(self, real1, real2, real3, number):
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
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.ser.write(voltage_message.encode())
        self.ser.flush()
        time.sleep(0.01)

    def reward_best_configurations(self, config, mse):
        """Store the best configurations based on MSE"""
        self.best_configurations.append((config, mse))
        self.best_configurations.sort(key=lambda x: x[1])  # Sort by MSE (ascending)
        if len(self.best_configurations) > 5:  # Keep top 5 configurations
            self.best_configurations = self.best_configurations[:5]

    def genetic_algorithm(self, config_manager, population_size=10, generations=5, mutation_rate=0.1):
        """Run a genetic algorithm to optimize configurations"""
        # Initialize population
        population = [config_manager.generate_random_config() for _ in range(population_size)]

        for generation in range(generations):
            print(f"Generation {generation}:")
            fitness = []

            for config in population:
                self.send_heater_values(config)
                time.sleep(2)
                # Measure outputs (dummy data used for fitness calculation)
                outputs = [random.uniform(0, 1) for _ in range(3)]  # Replace with oscilloscope.measure_outputs()
                mse = np.mean(outputs)  # Dummy MSE; replace with actual calculation
                fitness.append((config, mse))

            # Sort by fitness (lower MSE is better)
            fitness.sort(key=lambda x: x[1])
            self.best_configurations = fitness[:5]

            # Generate next generation
            next_population = []
            for i in range(0, len(fitness), 2):
                if i + 1 < len(fitness):
                    parent1, parent2 = fitness[i][0], fitness[i + 1][0]
                    child = config_manager.crossover(parent1, parent2)
                    child = config_manager.mutate(child, mutation_rate)
                    next_population.append(child)

            # Fill population to maintain size
            while len(next_population) < population_size:
                next_population.append(config_manager.generate_random_config())

            population = next_population

    def process_data(self, data_processor, oscilloscope, config_manager, num_loops=10):
        with open(data_processor.output_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Loop', 'MSE', 'Voltage'])

            for loop in range(num_loops):
                print(f"LOOP {loop} ------------------------")
                mse = self._process_loop(loop, data_processor, oscilloscope, config_manager, csvwriter)
                print(f"{loop} {mse}")

    def _process_loop(self, loop, data_processor, oscilloscope, config_manager, csvwriter):
        train, _ = train_test_split(data_processor.df['Id'], test_size=0.3)
        mse_temp = []

        for th in train:
            iris_data = data_processor.df.iloc[th]
            input_config = config_manager.generate_input_config(iris_data)
            self.send_heater_values(input_config)
            time.sleep(2)

            outputs = oscilloscope.measure_outputs()
            if outputs[0] is not None:
                mse = data_processor.calculate_mse(*outputs, th)
                mse_temp.append(mse)

        mse_avg = np.mean(mse_temp)
        config = config_manager.generate_random_config()
        self.reward_best_configurations(config, mse_avg)
        csvwriter.writerow([loop, mse_avg, config])
        return mse_avg

def main():
    oscilloscope = OscilloscopeController()
    config_manager = ConfigurationManager()
    data_processor = DataProcessor(
        csv_path='C:\\path\\to\\iris2.csv',
        output_path='C:\\path\\to\\output.csv'
    )
    serial_controller = SerialController()

    if not oscilloscope.connect_oscilloscope():
        print("Failed to connect to oscilloscope")
        return

    if not serial_controller.connect():
        print("Failed to connect to serial port")
        return

    data_processor.load_data()
    serial_controller.genetic_algorithm(config_manager)
    serial_controller.process_data(data_processor, oscilloscope, config_manager)

    print("Best Configurations:")
    for config, mse in serial_controller.best_configurations:
        print(f"MSE: {mse}, Config: {config}")

if __name__ == "__main__":
    main()
