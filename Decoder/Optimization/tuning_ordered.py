import serial
import time
import pyvisa
import numpy as np
import random
from typing import Dict, List, Tuple

# Constants
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
INPUT_HEATERS = [36, 37]
INPUT_STATES = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]
VOLTAGE_RANGE = (0.1, 4.9)

def setup_hardware() -> Tuple[serial.Serial, pyvisa.resources.Resource]:
    """Initialize and return serial and oscilloscope connections"""
    # Setup serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    
    # Setup oscilloscope
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    if not resources:
        raise Exception("No VISA resources found")
    
    scope = rm.open_resource(resources[0])
    scope.timeout = 5000
    
    # Configure oscilloscope channels
    for channel in range(1, 5):
        scope.write(f':CHANnel{channel}:DISPlay ON')
        scope.write(f':CHANnel{channel}:SCALe 2')
        scope.write(f':CHANnel{channel}:OFFSet -6')
    
    return ser, scope

def cleanup_hardware(ser: serial.Serial, scope: pyvisa.resources.Resource) -> None:
    """Clean up hardware connections"""
    ser.close()
    scope.close()

def send_voltages(ser: serial.Serial, config: Dict[str, float]) -> None:
    """Send voltage configuration to hardware"""
    voltage_message = "".join(f"{h},{v};" for h, v in config.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    time.sleep(0.25)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

def measure_channels(scope: pyvisa.resources.Resource) -> List[float]:
    """Measure voltage values from all channels"""
    outputs = []
    for channel in range(1, 5):
        try:
            value = float(scope.query(
                f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'
            ))
            outputs.append(round(value, 5))
        except Exception as e:
            print(f"Error measuring channel {channel}: {e}")
            outputs.append(None)
    return outputs

def evaluate_config(ser: serial.Serial, scope: pyvisa.resources.Resource, 
                   config: Dict[str, float], input_state: Tuple[float, float]) -> np.ndarray:
    """Evaluate a configuration with given input state"""
    test_config = config.copy()
    test_config[str(INPUT_HEATERS[0])] = input_state[0]
    test_config[str(INPUT_HEATERS[1])] = input_state[1]
    
    send_voltages(ser, test_config)
    outputs = measure_channels(scope)
    
    if None in outputs:
        return np.array([0.25, 0.25, 0.25, 0.25])  # Return uniform distribution on error
        
    # Convert to probabilities
    outputs_array = np.array(outputs)
    return outputs_array / np.sum(outputs_array)

def calculate_loss(probs: np.ndarray, target_idx: int) -> float:
    """Calculate loss for a single evaluation"""
    # Convert target to one-hot encoding

    target = np.zeros(4)
    target[target_idx] = 1
    
    # Cross entropy loss
    cross_entropy = -np.sum(target * np.log(probs + 1e-10))
    
    # Channel accuracy penalty
    actual_highest = int(np.argmax(probs))
    channel_penalty = 5.0 if actual_highest != target_idx else 0.0
    
    # Separation penalty
    sorted_probs = np.sort(probs)[::-1]
    separation_penalty = max(0, 0.3 - (sorted_probs[0] - sorted_probs[1])) * 3.0
    
    return cross_entropy + channel_penalty + separation_penalty

def optimize_spsa(ser: serial.Serial, scope: pyvisa.resources.Resource,
                 initial_config: Dict[str, float], 
                 iterations: int = 100,
                 delta: float = 0.1,
                 learning_rate: float = 0.01) -> Tuple[Dict[str, float], float]:
    """
    Optimize configuration using SPSA algorithm
    """
    current_config = initial_config.copy()
    best_config = current_config.copy()
    best_loss = float('inf')
    
    try:
        for iteration in range(iterations):
            # Decay learning rate
            current_lr = learning_rate / (1 + iteration * 0.1)
            
            # Generate perturbation vector
            delta_vector = {str(h): random.choice([-1, 1]) for h in range(40)}
            
            # Create perturbed configurations
            config_plus = {
                h: min(max(v + delta * delta_vector[h], VOLTAGE_RANGE[0]), VOLTAGE_RANGE[1])
                for h, v in current_config.items()
            }
            config_minus = {
                h: min(max(v - delta * delta_vector[h], VOLTAGE_RANGE[0]), VOLTAGE_RANGE[1])
                for h, v in current_config.items()
            }
            
            # Evaluate both configurations
            loss_plus = 0
            loss_minus = 0
            
            for idx, input_state in enumerate(INPUT_STATES):
                probs_plus = evaluate_config(ser, scope, config_plus, input_state)
                probs_minus = evaluate_config(ser, scope, config_minus, input_state)
                
                loss_plus += calculate_loss(probs_plus, idx)
                loss_minus += calculate_loss(probs_minus, idx)
            
            loss_plus /= len(INPUT_STATES)
            loss_minus /= len(INPUT_STATES)
            
            # Update parameters
            for heater in current_config.keys():
                gradient = (loss_plus - loss_minus) / (2 * delta * delta_vector[heater])
                new_value = current_config[heater] - current_lr * gradient
                current_config[heater] = min(max(new_value, VOLTAGE_RANGE[0]), VOLTAGE_RANGE[1])
            
            # Track best configuration
            current_loss = min(loss_plus, loss_minus)
            if current_loss < best_loss:
                best_loss = current_loss
                best_config = config_plus.copy() if loss_plus < loss_minus else config_minus.copy()
                print(f"New best loss: {best_loss:.4f} at iteration {iteration + 1}")
            
            if iteration % 5 == 0:
                print(f"Iteration {iteration + 1}: Current loss = {current_loss:.4f}")
                
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"Error during optimization: {e}")
    
    return best_config, best_loss

def format_config(config: Dict[str, float]) -> str:
    """Format configuration dictionary into a clean string"""
    # Sort by key and format each value to one decimal place
    items = [f"'{k}': {v:.1f}" for k, v in sorted(config.items(), key=lambda x: int(x[0]))]
    return '{' + ', '.join(items) + '}'


def main():
    """Main execution function"""
    try:
        # Initialize hardware
        ser, scope = setup_hardware()
        
        initial_config = {'0': 3.8, '1': 1.8, '2': 3.3, '3': 0.7, '4': 3.8, '5': 2.5, '6': 1.2, '7': 3.3, '8': 0.5, '9': 1.4, '10': 1.2, '11': 2.5, '12': 1.0, '13': 3.6, '14': 3.7, '15': 1.7, '16': 2.6, '17': 3.0, '18': 3.5, '19': 1.6, '20': 3.8, '21': 4.0, '22': 4.1, '23': 2.6, '24': 2.2, '25': 3.8, '26': 3.4, '27': 3.2, '28': 3.6, '29': 0.8, '30': 2.6, '31': 2.0, '32': 3.8, '33': 0.2, '34': 0.1, '35': 0.3, '36': 0.2, '37': 0.1, '38': 0.3, '39': 0.1}

        # Run optimization
        print("Starting optimization...")
        best_config, final_loss = optimize_spsa(
            ser=ser,
            scope=scope,
            initial_config=initial_config,
            iterations=10,
            delta=0.1,
            learning_rate=0.01
        )
        
        print("\nOptimization completed!")
        print(f"Final loss: {final_loss:.4f}")
        print(f"initial_config = {format_config(best_config)}")

            # Test final configuration with detailed analysis
        print("\nTesting final configuration:")
        for input_state in INPUT_STATES:
            current_config = best_config.copy()
            current_config[36] = input_state[0]
            current_config[37] = input_state[1]
            
            send_voltages(ser, current_config)
            time.sleep(0.25)
            outputs = measure_channels(scope)
            
            max_output = max(outputs)
            max_index = outputs.index(max_output)
            
            print(f"\nInputs (A, B): {input_state}")
            print(f"Outputs: {outputs}")
            print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
            other_outputs = outputs.copy()
            other_outputs.pop(max_index)
            print(f"Separation from next highest: {(max_output - max(other_outputs)):.4f}V")
    
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        # Always cleanup hardware
        cleanup_hardware(ser, scope)

if __name__ == "__main__":
    main()