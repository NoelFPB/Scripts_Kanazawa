import numpy as np
import time
import pyvisa as visa
import serial
import pandas as pd
from sklearn.model_selection import train_test_split
import random

class ConfigurationManager:
    """Manages configuration generation and perturbation"""
    def __init__(self):
        self.modifiable_heaters = sorted([i for i in range(35)], reverse=True)
        self.fixed_inputs = list(range(36, 40))
        self.voltage_min = 0.1
        self.voltage_max = 4.9

    def generate_random_config(self):
        """Generate a random configuration"""
        return {h: random.uniform(self.voltage_min, self.voltage_max) for h in self.modifiable_heaters}

    def generate_input_config(self, iris_data):
        """Create configuration for fixed inputs based on iris data"""
        return {
            36: iris_data['SeL'],  # Sepal Length
            37: iris_data['SeW'],  # Sepal Width
            38: iris_data['PeL'],  # Petal Length
            39: iris_data['PeW']   # Petal Width
        }

    def perturb_config(self, config, delta_vector, delta):
        """Apply perturbation to a configuration"""
        return {
            h: max(self.voltage_min, min(self.voltage_max, config[h] + delta * delta_vector[h]))
            for h in config
        }

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

    def calculate_loss(self, real, number):
        """Calculate cross-entropy loss"""
        target = self.target[number]
        real = np.array(real) / np.sum(real)  # Normalize outputs
        loss = -np.sum(np.array(target) * np.log(real + 1e-10))  # Add epsilon to avoid log(0)
        return loss

class SerialController:
    def __init__(self, port='COM4', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

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
        time.sleep(0.01)

    def train(self, config_manager, data_processor, oscilloscope, learning_rate=0.1, delta=0.05, iterations=500):
        """Train using SPSA"""
        print("\nStarting SPSA Training...")
        
        w = config_manager.generate_random_config()
        train_data = data_processor.train_indices

        for iteration in range(iterations):
            # Select a random training sample
            sample_id = random.choice(train_data)
            iris_data = data_processor.df.iloc[sample_id]

            # Generate fixed input configuration
            fixed_inputs = config_manager.generate_input_config(iris_data)

            # Generate random perturbation vector
            delta_vector = {h: random.choice([-1, 1]) for h in w}

            # Perturb configurations
            w_plus = config_manager.perturb_config(w, delta_vector, delta)
            w_minus = config_manager.perturb_config(w, delta_vector, -delta)

            # Combine trainable heaters with fixed inputs
            combined_plus = {**w_plus, **fixed_inputs}
            combined_minus = {**w_minus, **fixed_inputs}

            # Apply configurations and measure outputs
            self.send_heater_values(combined_plus)
            time.sleep(0.2)
            y_plus = oscilloscope.measure_outputs()

            self.send_heater_values(combined_minus)
            time.sleep(0.2)
            y_minus = oscilloscope.measure_outputs()

            if y_plus[0] is not None and y_minus[0] is not None:
                # Compute losses
                L_plus = data_processor.calculate_loss(y_plus, sample_id)
                L_minus = data_processor.calculate_loss(y_minus, sample_id)

                # Estimate gradients and update
                for h in w:
                    gradient = (L_plus - L_minus) / (2 * delta * delta_vector[h])
                    w[h] -= learning_rate * gradient

                # Clip values to valid range
                w = {
                    h: max(config_manager.voltage_min, min(config_manager.voltage_max, value))
                    for h, value in w.items()
                }

            if iteration % 50 == 0:
                print(f"Iteration {iteration}/{iterations}, Loss: {L_plus:.4f}")

        print("Training complete!")
        return w

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

    # Load data
    data_processor.load_data()

    # Train
    best_config = serial_controller.train(
        config_manager=config_manager,
        data_processor=data_processor,
        oscilloscope=oscilloscope,
        learning_rate=0.1,
        delta=0.05,
        iterations=500
    )

    print("Best Configuration:", best_config)

if __name__ == "__main__":
    main()
