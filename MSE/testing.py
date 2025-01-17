import numpy as np
import time
import pyvisa as visa
import serial
import pandas as pd
from sklearn.model_selection import train_test_split

class ConfigurationManager:
    """Manages configuration for fixed inputs"""
    def __init__(self):
        self.fixed_inputs = list(range(36, 40))

    def generate_input_config(self, iris_data):
        """Create configuration for fixed inputs based on iris data"""
        return {
            36: iris_data['SeL'],
            37: iris_data['SeW'],
            38: iris_data['PeL'],
            39: iris_data['PeW']
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
        self.test_indices = None

    def load_data(self):
        """Load data and create test split"""
        self.df = pd.read_csv(self.csv_path)
        _, self.test_indices = train_test_split(
            self.df['Id'].apply(lambda x: x - 1), 
            test_size=0.3, 
            random_state=42
        )

class SerialController:
    def __init__(self, port='COM4', baudrate=9600):
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

    def test_fixed_configuration(self, data_processor, oscilloscope, config_manager):
        """Testing phase using a pre-defined heater configuration"""
        print("\nStarting Testing with Fixed Configuration...")
        
        test_indices = [73, 18, 118, 78, 76, 31, 64, 141, 68, 82, 110, 12, 36, 9, 19, 
                       56, 104, 69, 55, 132, 29, 127, 26, 128, 131, 145, 108, 143, 45, 
                       30, 22, 15, 65, 11, 42, 146, 51, 27, 4, 32, 142, 85, 86, 16, 10]
        
        # best_config = {34: 2.0, 33: 0.5, 32: 1.0, 31: 2.0, 30: 1.0, 29: 0.5, 28: 3.5, 
        #               27: 0.1, 26: 2.5, 25: 4.0, 24: 3.0, 23: 0.5, 22: 4.9, 21: 2.0, 
        #               20: 2.5, 19: 4.9, 18: 2.5, 17: 4.5, 16: 4.0, 15: 0.5, 14: 4.5, 
        #               13: 3.5, 12: 1.5, 11: 2.5, 10: 3.0, 9: 3.5, 8: 4.9, 7: 2.5, 
        #               6: 0.1, 5: 1.5, 4: 3.5, 3: 2.5, 2: 3.5, 1: 0.1, 0: 3.5}
        
        #best_config = {0: 3.0, 1: 2.0, 2: 1.5, 3: 0.5, 4: 2.0, 5: 4.0, 6: 4.9, 7: 2.5, 8: 2.0, 9: 3.0, 10: 1.0, 11: 4.5, 12: 2.5, 13: 0.1, 14: 0.1, 15: 2.0, 16: 4.9, 17: 1.5, 18: 4.0, 19: 4.0, 20: 4.5, 21: 1.5, 22: 3.0, 23: 3.5, 24: 2.0, 25: 2.0, 26: 4.9, 27: 2.0, 28: 3.5, 29: 4.5, 30: 1.0, 31: 2.0, 32: 4.5, 33: 1.0, 34: 1.0}
        best_config ={0: 4.5, 1: 2.0, 2: 1.0, 3: 1.0, 4: 3.0, 5: 3.0, 6: 2.5, 7: 0.1, 8: 4.0, 9: 2.5, 10: 2.5, 11: 2.5, 12: 0.5, 13: 4.0, 14: 2.0, 15: 0.1, 16: 3.0, 17: 1.0, 18: 0.1, 19: 1.0, 20: 4.5, 21: 2.0, 22: 0.1, 23: 2.5, 24: 4.9, 25: 4.5, 26: 3.0, 27: 3.5, 28: 3.0, 29: 2.0, 30: 2.5, 31: 4.0, 32: 3.5, 33: 3.5, 34: 4.9}
        confusion_matrix = np.zeros((3, 3), dtype=int)
        classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        
        self.send_heater_values(best_config)
        time.sleep(2)
            
        for sample_id in test_indices:
            iris_data = data_processor.df.iloc[sample_id]
            input_config = config_manager.generate_input_config(iris_data)
            
            actual_class = classes.index(iris_data['Species'])
            
            self.send_heater_values(input_config)
            time.sleep(0.2)
            # With 0.2 seconds I get the same result as with 2 seconds.
            # The scale shouuld be as one block 200uS
            outputs = oscilloscope.measure_outputs()
            print(f"Sample outputs: {outputs}")
            
            if outputs[0] is not None:
                outputs_array = np.array(outputs)
                output_probs = outputs_array / np.sum(outputs_array)
                predicted_class = np.argmax(output_probs)
                confusion_matrix[actual_class][predicted_class] += 1
                print(f"Sample {sample_id} - True: {classes[actual_class]}, Predicted: {classes[predicted_class]}")
        
        print("\nConfusion Matrix:")
        print("Predicted →")
        print("Actual ↓")
        print("            Setosa  Versicolor  Virginica")
        for i, actual in enumerate(classes):
            print(f"{actual:12} {confusion_matrix[i][0]:7d} {confusion_matrix[i][1]:11d} {confusion_matrix[i][2]:9d}")
        
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        print(f"\nOverall Accuracy: {accuracy:.2%}")

def main():
    oscilloscope = OscilloscopeController()
    oscilloscope.initialize_channels()
    config_manager = ConfigurationManager()
    data_processor = DataProcessor('C:\\Users\\noelp\\Documents\\Kanazawa\\Scripts_Kanazawa\\MSE\\Datasets\\iris_normalized.csv')
    serial_controller = SerialController()

    if not oscilloscope.connect_oscilloscope():
        print("Failed to connect to oscilloscope")
        return
    if not serial_controller.connect():
        print("Failed to connect to serial port")
        return

    data_processor.load_data()
    
    serial_controller.test_fixed_configuration(
        data_processor=data_processor,
        oscilloscope=oscilloscope,
        config_manager=config_manager
    )

if __name__ == "__main__":
    main()