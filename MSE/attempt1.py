import numpy as np
import time
import csv
import pyvisa as visa
import serial
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class OscilloscopeController:
    def __init__(self):
        # Oscilloscope configuration
        self.channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
        self.scope = None
        
    def connect_oscilloscope(self):
        """Connect to the oscilloscope"""
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
        """Initialize oscilloscope channels"""
        if not self.scope:
            return
            
        for channel in self.channels:
            self.scope.write(f':{channel}:DISPlay ON')   # activate channels
            self.scope.write(f':{channel}:SCALe 2')      # scale
            self.scope.write(f':{channel}:OFFSet -6')    # offset

    def measure_outputs(self):
        """Measure outputs from oscilloscope"""
        try:
            outputs = []
            for channel in range(1, 4):  # Only channels 1-3 for MSE calculation
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
        """Load and process the iris dataset"""
        self.df = pd.read_csv(self.csv_path)
        self._print_data()
        self._create_target_encoding()
        
    def _print_data(self):
        """Print formatted data"""
        print('{:<10}{:<10}{:<10}{:<10}{:<10}{:<20}'.format(
            'Id', 'SeL', 'SeW', 'PeL', 'PeW', 'Species'))
        
        for i in range(len(self.df)):
            print('{:<10}{:<10}{:<10}{:<10}{:<10}{:<20}'.format(
                self.df['Id'][i],
                self.df['SeL'][i],
                self.df['SeW'][i],
                self.df['PeL'][i],
                self.df['PeW'][i],
                self.df['Species'][i]))
                
    def _create_target_encoding(self):
        """Create one-hot encoding for species"""
        for species in self.df['Species']:
            if species == "Iris-setosa":
                self.target.append([1,0,0])
            elif species == "Iris-versicolor":
                self.target.append([0,1,0])
            elif species == "Iris-virginica":
                self.target.append([0,0,1])
    
    def calculate_mse(self, real1, real2, real3, number):
        """Calculate MSE for given outputs"""
        target = self.target[number]
        real = np.array([real1, real2, real3])
        ratio = real / np.sum(real)
        mse = (np.sum((target - ratio)**2)/3)
        return round(mse, 5)

class SerialController:
    def __init__(self, port='COM4', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        
    def connect(self):
        """Establish serial connection"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate)
            time.sleep(2)  # Wait for connection to stabilize
            return True
        except Exception as e:
            print(f"Failed to connect to serial port: {e}")
            return False
            
    def send_heater_values(self, config):
        """Send heater values via serial connection"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.ser.write(voltage_message.encode())
        self.ser.flush()
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(0.01)

    def initialize_output(self):
        """Initialize output string"""
        config = {i: 0 for i in range(40)}  # Initialize all heaters to 0
        self.send_heater_values(config)
        time.sleep(1)
        
    def process_data(self, data_processor, oscilloscope, num_loops=10):
        """Process data through serial connection with MSE calculation"""
        with open(data_processor.output_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Loop', 'MSE', 'Voltage', 'Test'])
            
            for loop in range(num_loops):
                print(f"LOOP {loop} ------------------------")
                mse = self._process_loop(loop, data_processor, oscilloscope, csvwriter)
                print(f"{loop} {mse}")
                
    def _process_loop(self, loop, data_processor, oscilloscope, csvwriter):
        """Process a single loop of data"""
        train, test = train_test_split(data_processor.df['Id'], test_size=0.3)
        mse_temp = []
        
        # Generate and send weight voltages
        weight = np.round(np.random.uniform(0, 5, 35), 2)
        config = {i: w for i, w in enumerate(weight)}  # Create config dictionary
        self.send_heater_values(config)
        time.sleep(2)
        
        # Process training data
        for th in train:
            iris_data = data_processor.df.iloc[th]
            # Create config dictionary for iris data
            iris_config = {
                36: iris_data['SeL'],
                37: iris_data['SeW'],
                38: iris_data['PeL'],
                39: iris_data['PeW']
            }
            self.send_heater_values(iris_config)
            time.sleep(2)  # Wait for system to stabilize
            
            # Read oscilloscope data and calculate MSE
            outputs = oscilloscope.measure_outputs()
            if outputs[0] is not None:
                mse = data_processor.calculate_mse(*outputs, th)
                mse_temp.append(mse)
        
        # Calculate average MSE for this loop
        mse_avg = np.mean(mse_temp)
        csvwriter.writerow([loop, mse_avg, weight.tolist(), test.tolist()])
        return mse_avg

def main():
    # Initialize controllers
    oscilloscope = OscilloscopeController()
    data_processor = DataProcessor(
        csv_path='C:\\Users\\noelp\\Documents\\Kanazawa\\Scripts_Kanazawa\\MSI\\iris2.csv',
        output_path='C:\\Users\\noelp\\Documents\\Kanazawa\\Scripts_Kanazawa\\MSI\\large_scale.csv'
    )
    serial_controller = SerialController()
    
    # Connect to devices
    if not oscilloscope.connect_oscilloscope():
        print("Failed to connect to oscilloscope")
        return
        
    if not serial_controller.connect():
        print("Failed to connect to serial port")
        return
    
    # Load and process data
    data_processor.load_data()
    
    # Initialize serial output
    serial_controller.initialize_output()
    
    # Process data through serial connection
    serial_controller.process_data(data_processor, oscilloscope)

if __name__ == "__main__":
    main()