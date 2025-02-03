import numpy as np
import time
import pyvisa
import serial
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

class ConfigurationManager:
    """Manages configuration generation and perturbation"""
    def __init__(self):
        self.modifiable_heaters = sorted([i for i in range(35)], reverse=True)
        self.fixed_inputs = list(range(36, 39))  # Only 3 inputs for decoder (A, B, C)
        self.voltage_min = 0.1
        self.voltage_max = 4.9

    def generate_random_config(self):
        """Generate a random configuration"""
        return {h: random.uniform(self.voltage_min, self.voltage_max) 
                for h in self.modifiable_heaters}

    def generate_input_config(self, decoder_data):
        """Create configuration for fixed inputs based on decoder data"""
        return {
            36: decoder_data['A'],
            37: decoder_data['B'],
            38: decoder_data['C']
        }

    def perturb_config(self, config, delta_vector, delta):
        """Apply perturbation to a configuration"""
        return {
            h: max(self.voltage_min, min(self.voltage_max, 
                  config[h] + delta * delta_vector[h]))
            for h in config
        }
class OscilloscopeController:
    def __init__(self):
        self.scopes = None
        self.SCOPE1_ID = 'USB0::0x1AB1::0x0610::HDO1B244000779::INSTR'
        self.SCOPE2_ID = 'USB0::0x1AB1::0x0610::HDO1B244100809::INSTR'
        
    def init_scopes(self):
        """Initialize both oscilloscopes"""
        try:
            rm = pyvisa.ResourceManager()
            scope1 = rm.open_resource(self.SCOPE1_ID)
            scope2 = rm.open_resource(self.SCOPE2_ID)
            self.scopes = [scope1, scope2]
            
            for scope in self.scopes:
                scope.timeout = 5000
                for i in range(1, 5):
                    scope.write(f':CHANnel{i}:DISPlay ON')
                    scope.write(f':CHANnel{i}:SCALe 2')
                    scope.write(f':CHANnel{i}:OFFSet -6')
            return True
        except Exception as e:
            print(f"Failed to initialize scopes: {e}")
            return False
            
    def measure_scope(self, scope_idx, output_queue):
        """Measure outputs from a single scope"""
        try:
            scope = self.scopes[scope_idx]
            outputs = []
            # First scope has 4 channels, second has 3
            num_channels = 4 if scope_idx == 0 else 3
            
            for channel in range(1, num_channels + 1):
                value = float(scope.query(
                    f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
                outputs.append(round(value, 5))
            
            output_queue.put((scope_idx, outputs))
            
        except Exception as e:
            print(f"Error measuring scope {scope_idx}: {e}")
            output_queue.put((scope_idx, [None] * (4 if scope_idx == 0 else 3)))

    def measure_outputs(self):
        """Measure outputs from both oscilloscopes in parallel"""
        try:
            output_queue = Queue()
            threads = []
            
            # Start measurement threads for both scopes
            for scope_idx in range(2):
                thread = threading.Thread(
                    target=self.measure_scope,
                    args=(scope_idx, output_queue)
                )
                threads.append(thread)
                thread.start()
                
            # Wait for both threads to complete
            for thread in threads:
                thread.join()
                
            # Collect results
            all_outputs = []
            scope_results = {}
            
            # Get results from queue
            while not output_queue.empty():
                scope_idx, outputs = output_queue.get()
                scope_results[scope_idx] = outputs
                
            # Combine results in correct order
            if 0 in scope_results and 1 in scope_results:
                all_outputs.extend(scope_results[0])  # First scope (4 channels)
                all_outputs.extend(scope_results[1])  # Second scope (3 channels)
                return all_outputs
            else:
                print("Error: Missing results from one or both scopes")
                return [None] * 7
                
        except Exception as e:
            print(f"Measurement error: {e}")
            return [None] * 7
    
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
        """Create one-hot encoding for decoder outputs"""
        for output in self.df['Output']:
            # Create 8-element array with 1 at the correct output position
            target = [0] * 7
            output_num = int(output.replace('Out', ''))
            target[output_num] = 1
            self.target.append(target)

    def calculate_loss(self, real, number):
        """Calculate cross-entropy loss"""
        target = self.target[number]
        real = np.array(real) / np.sum(real)  # Normalize outputs
        loss = -np.sum(np.array(target) * np.log(real + 1e-10))
        return loss

    def format_config(self, config):
        """Convert config dictionary to clean float values"""
        return {k: round(float(v), 2) for k, v in config.items()}

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
        voltage_message = "".join(
            f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.ser.write(voltage_message.encode())
        self.ser.flush()
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(0.01)

    def evaluate_config(self, config, validation_data, data_processor, 
                       config_manager, oscilloscope):
        """Evaluate configuration on validation set"""
        correct = 0
        total = 0
        
        for sample_id in validation_data:
            decoder_data = data_processor.df.iloc[sample_id]
            input_config = config_manager.generate_input_config(decoder_data)
            combined_config = {**config, **input_config}
            
            self.send_heater_values(combined_config)
            time.sleep(0.25)
            
            outputs = oscilloscope.measure_outputs()
            if outputs[0] is not None:
                outputs_array = np.array(outputs)
                output_probs = outputs_array / np.sum(outputs_array)
                predicted_class = np.argmax(output_probs)
                actual_class = np.argmax(data_processor.target[sample_id])
                
                if predicted_class == actual_class:
                    correct += 1
                total += 1
                
        return correct / total if total > 0 else 0

    def test_configuration(self, config, data_processor, oscilloscope, config_manager):
        """Testing phase using the trained configuration"""
        print("\nStarting Testing with Trained Configuration...")
        confusion_matrix = np.zeros((7, 7), dtype=int)  # 8x8 for decoder outputs
        
        # First apply the trained heater configuration
        self.send_heater_values(config)
        time.sleep(2)  # Let system stabilize with base configuration
        
        for sample_id in data_processor.test_indices:
            decoder_data = data_processor.df.iloc[sample_id]
            input_config = config_manager.generate_input_config(decoder_data)
            actual_class = int(decoder_data['Output'].replace('Out', ''))
            
            self.send_heater_values(input_config)
            time.sleep(0.25)
            
            outputs = oscilloscope.measure_outputs()
            print(f"Sample outputs: {outputs}")
            
            if outputs[0] is not None:
                outputs_array = np.array(outputs)
                output_probs = outputs_array / np.sum(outputs_array)
                predicted_class = np.argmax(output_probs)
                confusion_matrix[actual_class][predicted_class] += 1
                print(f"Sample {sample_id} - True: Out{actual_class}, "
                      f"Predicted: Out{predicted_class}")
        
        print("\nConfusion Matrix:")
        print(" " * 10 + "Predicted →")
        print("Actual ↓  " + "".join(f"Out{i:1d}    " for i in range(7)))
        for i in range(7):
            print(f"Out{i:1d}      " + "".join(f"{n:5d} " for n in confusion_matrix[i]))
        
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        return None

    def fine_tune(self, initial_config, config_manager, data_processor, oscilloscope,
                learning_rate=0.05, iterations=30, patience=10):
        """Fine-tune the configuration using a more focused approach"""
        print("\nStarting Fine-tuning Phase...")
        best_config = initial_config.copy()
        best_accuracy = 0
        patience_counter = 0
        w = initial_config.copy()
        
        # Create validation set from training data
        train_data = list(data_processor.train_indices)
        random.shuffle(train_data)
        validation_size = int(len(train_data) * 0.2)
        validation_data = train_data[:validation_size]
        
        for iteration in range(iterations):
            # Select a subset of heaters to tune
            heaters_to_tune = random.sample(config_manager.modifiable_heaters, k=5)
            
            for heater in heaters_to_tune:
                # Try small perturbations in both directions
                deltas = [-0.3, 0.3]
                best_delta = 0
                best_local_accuracy = 0
                
                for delta in deltas:
                    temp_config = w.copy()
                    temp_config[heater] = np.clip(
                        temp_config[heater] + delta,
                        config_manager.voltage_min,
                        config_manager.voltage_max
                    )
                    
                    accuracy = self.evaluate_config(
                        temp_config,
                        validation_data,
                        data_processor,
                        config_manager,
                        oscilloscope
                    )
                    
                    if accuracy > best_local_accuracy:
                        best_local_accuracy = accuracy
                        best_delta = delta
                
                # Apply the best perturbation
                w[heater] = np.clip(
                    w[heater] + best_delta,
                    config_manager.voltage_min,
                    config_manager.voltage_max
                )
            
            # Evaluate current configuration
            current_accuracy = self.evaluate_config(
                w,
                validation_data,
                data_processor,
                config_manager,
                oscilloscope
            )
            
            print(f"Fine-tuning iteration {iteration}/{iterations}, "
                  f"Validation Accuracy: {current_accuracy:.2%}")
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_config = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {iteration + 1} iterations")
                break
        
        print(f"Fine-tuning complete! Best validation accuracy: {best_accuracy:.2%}")
        return best_config

    def train(self, config_manager, data_processor, oscilloscope,
             learning_rate=0.1, delta=0.5, iterations=300):
        """Complete training process: SPSA followed by fine-tuning"""
        # SPSA Training
        print("\nStarting SPSA Training...")
        w = config_manager.generate_random_config()
        train_data = list(data_processor.train_indices)
        
        for iteration in range(iterations):
            # Select a random training sample
            sample_id = random.choice(train_data)
            decoder_data = data_processor.df.iloc[sample_id]
            
            # Generate fixed input configuration
            fixed_inputs = config_manager.generate_input_config(decoder_data)
            
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
            time.sleep(0.25)
            y_plus = oscilloscope.measure_outputs()
            
            self.send_heater_values(combined_minus)
            time.sleep(0.25)
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
                w = {h: np.clip(value, config_manager.voltage_min, config_manager.voltage_max)
                     for h, value in w.items()}
                
                print(f"Iteration {iteration}/{iterations}, Loss: {L_plus:.4f}")
        
        print("SPSA Training complete!")
        
        # Fine-tuning phase
        final_config = self.fine_tune(
            initial_config=w,
            config_manager=config_manager,
            data_processor=data_processor,
            oscilloscope=oscilloscope
        )
        
        return final_config

def main():
    oscilloscope = OscilloscopeController()
    config_manager = ConfigurationManager()
    data_processor = DataProcessor('C:\\Users\\noelp\\Documents\\Kanazawa\\Scripts_Kanazawa\\Decoder\\3_bits\\Datasets\\decoder_normalized.csv')
    serial_controller = SerialController()

    # Setup connections
    if not oscilloscope.init_scopes():
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
        delta=0.5,
        iterations=300
    )

    clean_config = data_processor.format_config(best_config)
    print("Best Configuration:", clean_config)

    # Test the trained configuration
    serial_controller.test_configuration(
        config=best_config,
        data_processor=data_processor,
        oscilloscope=oscilloscope,
        config_manager=config_manager)

if __name__ == "__main__":
    main()
