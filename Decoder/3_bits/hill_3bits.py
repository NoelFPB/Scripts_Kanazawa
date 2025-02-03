import serial
import time
import pyvisa
import random

# Serial port configuration 
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Initialize oscilloscopes
rm = pyvisa.ResourceManager()
resources = rm.list_resources()
if len(resources) < 2:
    raise Exception("Need at least 2 VISA resources for oscilloscopes")

# Define specific scope resources
SCOPE1_ID = 'USB0::0x1AB1::0x0610::HDO1B244000779::INSTR'  # First scope (4 channels)
SCOPE2_ID = 'USB0::0x1AB1::0x0610::HDO1B244100809::INSTR'  # Second scope (3 channels)

# Connect to specific scopes
scope1 = rm.open_resource(SCOPE1_ID)
scope2 = rm.open_resource(SCOPE2_ID)
scopes = [scope1, scope2]

# Print confirmation
print(f"Scope 1 (4 channels): {scope1.query('*IDN?').strip()}")
print(f"Scope 2 (3 channels): {scope2.query('*IDN?').strip()}")

for scope in scopes:
    scope.timeout = 5000

# Setup channels for both scopes
def setup_scope_channels(scope):
    channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
    for channel in channels:
        scope.write(f':{channel}:DISPlay ON')
        scope.write(f':{channel}:SCALe 2')
        scope.write(f':{channel}:OFFSet -6')

for scope in scopes:
    setup_scope_channels(scope)

# Heater configuration
heater_values = {i: 0.0 for i in range(40)}
fixed_first_layer = list(range(33, 40))
input_heaters = [35, 36, 37]  # Now using 3 input heaters
# Generate all 8 combinations for 3 inputs
input_combinations = [(v1, v2, v3) for v1 in [0.1, 4.9] 
                                  for v2 in [0.1, 4.9] 
                                  for v3 in [0.1, 4.9]]
modifiable_heaters = [i for i in range(33) if i not in input_heaters]

# Set fixed first layer
for heater in fixed_first_layer:
    heater_values[heater] = 0.01

# Voltage options - simplified set for faster convergence
voltage_options = [0.1, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.9]

def send_heater_values(ser, config):
    """Send heater values via serial connection"""
    voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.01)

def measure_outputs():
    """Measure outputs from both oscilloscopes"""
    try:
        outputs = []
        # First scope - all 4 channels
        for channel in range(1, 5):
            value = float(scope1.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
            outputs.append(round(value, 5))
        # Second scope - first 3 channels only
        for channel in range(1, 4):
            value = float(scope2.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel}'))
            outputs.append(round(value, 5))
        return outputs
    except Exception as e:
        print(f"Measurement error: {e}")
        return [None] * 7

def evaluate_configuration(ser, config):
    """Evaluate a single configuration for 3-bit decoder behavior"""
    total_score = 0

    print("\n" + "="*50)
    #print("Current configuration evaluation:")
    
    # Expected highest output for each input combination (3-bit)
    expected_outputs = {
        (0.1, 0.1, 0.1): 0,  # 000
        (0.1, 0.1, 4.9): 1,  # 001
        (0.1, 4.9, 0.1): 2,  # 010
        (0.1, 4.9, 4.9): 3,  # 011
        (4.9, 0.1, 0.1): 4,  # 100
        (4.9, 0.1, 4.9): 5,  # 101
        (4.9, 4.9, 0.1): 6   # 110
        # Note: 111 is not used as we only have 7 outputs
    }
    
    for input_state in input_combinations:
        # Skip the last combination (111) as we only have 7 outputs
        if input_state == (4.9, 4.9, 4.9):
            continue
            
        current_config = config.copy()
        current_config[35] = input_state[0]
        current_config[36] = input_state[1]
        current_config[37] = input_state[2]
        
        send_heater_values(ser, current_config)
        time.sleep(0.25)
        
        outputs = measure_outputs()
        if None in outputs:
            return -1000, []
       # Print current readings
        #print("\n             Scope 1                 |    Scope 2")
        print(f"Input {input_state}: ", end="")
        # Print all readings on one line
        for i in range(7):
            if i == 4:
                print(" | ", end="")
            print(f"{outputs[i]:6.4f}V ", end="")
        print()  # New line at the end
        
        # Find highest output
        max_output = max(outputs)
        actual_highest = outputs.index(max_output)
        expected_highest = expected_outputs[input_state]
    
        
        # Score based on correct output being highest
        if actual_highest == expected_highest:
            # Additional points for separation
            other_outputs = outputs.copy()
            other_outputs.pop(actual_highest)
            separation = max_output - max(other_outputs)
            total_score += 10 + min(separation * 5, 10)  # Up to 10 bonus points for separation
        
    
    return total_score

def hill_climbing_optimization(ser, iterations=20):
    """Hill climbing optimization for 3-bit decoder"""
    # Initialize with random configuration
    best_config = {str(i): random.choice(voltage_options) for i in modifiable_heaters}
    for heater in fixed_first_layer:
        best_config[str(heater)] = 0.01
    
    best_score = evaluate_configuration(ser, best_config)
    print(f"Initial score: {best_score}")
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}/{iterations}")
        improved = False
        
        # Try to improve each heater
        for heater in modifiable_heaters:
            current_value = best_config[str(heater)]
            
            # Try different values
            for new_value in voltage_options:
                if new_value != current_value:
                    test_config = best_config.copy()
                    test_config[str(heater)] = new_value
                    
                    score = evaluate_configuration(ser, test_config)
                    if score > best_score:
                        best_score = score
                        best_config = test_config.copy()
                        improved = True
                        #print(f"Improved score: {best_score}")
                        break
                    print(f"Current score {score} Best score {best_score}")
                    
            if improved:
                break
        
        if not improved:
            print("No improvement found in this iteration")
            break
    
    return best_config, best_score


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(0.5)
        
        print("Starting optimization for 3-bit decoder...")
        best_config, best_score = hill_climbing_optimization(ser)
        
        print("\nOptimization Complete!")
        print(f"Best score: {best_score}")
        
        # Print final heater configuration
        print("\nFinal Heater Configuration:")
        for heater in sorted(best_config.keys()):
            print(f"Heater {heater}: {best_config[heater]:.2f}V")
        
        # Test final configuration with detailed analysis
        print("\nTesting final configuration:")
        for input_state in input_combinations:
            # Skip 111 combination
            if input_state == (4.9, 4.9, 4.9):
                continue
                
            current_config = best_config.copy()
            current_config[35] = input_state[0]
            current_config[36] = input_state[1]
            current_config[37] = input_state[2]
            
            send_heater_values(ser, current_config)
            time.sleep(0.25)
            outputs = measure_outputs()
            
            max_output = max(outputs)
            max_index = outputs.index(max_output)
            
            print(f"\nInputs (A, B, C): {input_state}")
            print(f"Outputs: {outputs}")
            print(f"Highest output: Channel {max_index + 1} = {max_output:.4f}V")
            other_outputs = outputs.copy()
            other_outputs.pop(max_index)
            print(f"Separation from next highest: {(max_output - max(other_outputs)):.4f}V")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        ser.close()
        scope1.close()
        scope2.close()

if __name__ == "__main__":
    main()