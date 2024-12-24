import serial
import time
import pyvisa
import random
import json
import pandas as pd

# Same as main training, but instead of tryig to test multiple configurations, once it founds one that could work as a decoder it iterates over it.

# Serial port configuration
SERIAL_PORT = 'COM4'  # Replace with your serial port
BAUD_RATE = 9600

# Initialize oscilloscope
rm = pyvisa.ResourceManager()
resources = rm.list_resources()
if not resources:
    raise Exception("No VISA resources found. Make sure the oscilloscope is connected.")
scope = rm.open_resource(resources[0])
scope.timeout = 5000

# Initial oscilloscope setup
channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
for channel in channels:
    scope.write(f':{channel}:DISPlay ON')
    scope.write(f':{channel}:SCALe 2')
    scope.write(f':{channel}:OFFSet -6')

# Heater values for all 40 heaters (reversed order)
heater_values = {i: 0.0 for i in range(40)}

# Fixed first layer (heaters 33-39)
fixed_first_layer_heaters = list(range(33, 40))
for heater in fixed_first_layer_heaters:
    heater_values[heater] = 0.01

# Input heaters (36 and 37)
input_heaters = [36, 37]
input_combinations = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]

# Modifiable heaters (all except 33-39)
modifiable_heaters = [i for i in range(40) if i not in fixed_first_layer_heaters]

# Voltage range for modifiable heaters
voltage_range = [0.1, 1.0, 2.0, 3.0, 4.0, 4.9]

# Function to send heater values
def send_heater_values(ser):
    voltage_message = "".join(f"{heater},{value};" for heater, value in heater_values.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.01)

# Function to measure outputs
def measure_outputs():
    try:
        output1 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel1')), 5)
        output2 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel2')), 5)
        output3 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel3')), 5)
        output4 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel4')), 5)
        return [output1, output2, output3, output4]
    except Exception as e:
        print(f"Error measuring outputs: {e}")
        return [None, None, None, None]

# Function to check decoder behavior
def check_decoder_behavior(df):
    """
    For each input state, checks if exactly one output is at least 1V higher than the others.
    Returns True if this condition holds for all input states, along with HeaterConfiguration.
    """
    for _, row in df.iterrows():
        outputs = [row['Output1'], row['Output2'], row['Output3'], row['Output4']]
        max_output = max(outputs)
        distinct_count = sum(1 for o in outputs if max_output - o >= 1.0)
        if distinct_count != 3:  # Three outputs must be at least 1V lower
            return False, None
    return True, df.iloc[0]['HeaterConfiguration']

# Function to check unique dominance
def check_unique_dominance(group, threshold=1.0):
    results = []
    dominant_bits = {}

    for index, row in group.iterrows():
        # Get the input state and outputs
        input_state = (row['InputState_Heater36'], row['InputState_Heater37'])
        outputs = [row['Output1'], row['Output2'], row['Output3'], row['Output4']]
        max_val = max(outputs)
        # Identify the index of the dominant output
        dominant_index = outputs.index(max_val)
        # Ensure the dominant output is significantly higher than the rest
        differences = [max_val - output for output in outputs]
        distinctly_higher = sum(diff > threshold for diff in differences) == (len(outputs) - 1)

        if distinctly_higher:
            # Check if this dominant index is already associated with another input state
            if dominant_index in dominant_bits.values():
                print(f"Conflict found: Input state {input_state} produces the same dominant output as another row.")
                return False  # Conflict found, not decoder-like behavior
            # Otherwise, save the dominant index for this input state
            dominant_bits[input_state] = dominant_index
            results.append((index, input_state, dominant_index))

    return results

# Function to calculate decoder-like behavior score
def calculate_decoder_score(df, threshold=1.0):
    """
    Calculate a score to evaluate how well a configuration exhibits decoder-like behavior.
    Lower scores indicate better behavior.
    """
    total_penalty = 0
    for _, row in df.iterrows():
        outputs = [row['Output1'], row['Output2'], row['Output3'], row['Output4']]
        max_output = max(outputs)
        outputs.remove(max_output)
        next_highest = max(outputs)

        # Calculate penalty
        penalty = max(0, threshold - (max_output - next_highest))
        total_penalty += penalty

    # Normalize by the number of input states
    return total_penalty / len(df)

# Main function
def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)

    print("Searching for a Decoder-Like Configuration...")
    refinement_iterations = 400  # Number of refinement iterations
    decoder_found = False

    initial_config = None  # Initialize initial_config outside the loop
    while not decoder_found:
        # Randomly set a configuration
        for heater in modifiable_heaters:
            heater_values[heater] = random.choice(voltage_range)

        config_results = []

        for input_state in input_combinations:
            heater_values[36], heater_values[37] = input_state
            send_heater_values(ser)

            outputs = measure_outputs()
            if outputs == [None, None, None, None]:
                continue

            heater_config_json = json.dumps(heater_values)
            config_results.append({
                'InputState_Heater36': input_state[0],
                'InputState_Heater37': input_state[1],
                'Output1': outputs[0],
                'Output2': outputs[1],
                'Output3': outputs[2],
                'Output4': outputs[3],
                'HeaterConfiguration': heater_config_json
            })

            print(f"InputState: {input_state}, Outputs: {outputs}, HeaterConfig: {heater_config_json}")

        df = pd.DataFrame(config_results)

        # Check for basic decoder behavior
        decoder_found, initial_config_json = check_decoder_behavior(df)
        if decoder_found:
            # Check for unique dominance
            dominance_results = check_unique_dominance(df)
            if not dominance_results:  # Conflict found, skip this configuration
                print("Conflict detected in dominance. Skipping this configuration...")
                decoder_found = False
                continue

            print("Decoder-like configuration with unique dominance found!")
            initial_config = json.loads(initial_config_json)

            # Validate and complete initial_config
            for heater in modifiable_heaters:
                if heater not in initial_config:
                    initial_config[heater] = 0.1  # Default to the minimum voltage
            break

    # This if could be deleted
    if initial_config is None:
        print("No valid decoder-like configuration found. Exiting...")
        ser.close()
        return
    
    best_score = float('inf')
    best_config = initial_config.copy()
    no_improvement_count = 0  # Counter to track consecutive failures to improve

    for iteration in range(refinement_iterations):
        print(f"Iteration {iteration + 1}...")

        # Slightly adjust the configuration
        for heater in modifiable_heaters:
            initial_config[heater] = max(0.1, min(4.9, initial_config[heater] + random.uniform(-0.1, 0.1)))

        iteration_results = []

        for input_state in input_combinations:
            initial_config[36], initial_config[37] = input_state
            send_heater_values(ser)

            outputs = measure_outputs()
            if outputs == [None, None, None, None]:
                continue

            iteration_results.append({
                'Iteration': iteration + 1,
                'InputState_Heater36': input_state[0],
                'InputState_Heater37': input_state[1],
                'Output1': outputs[0],
                'Output2': outputs[1],
                'Output3': outputs[2],
                'Output4': outputs[3],
                'HeaterConfiguration': json.dumps(initial_config)
            })

        refined_df = pd.DataFrame(iteration_results)

        # Check score and unique dominance
        score = calculate_decoder_score(refined_df)
        dominance_results = check_unique_dominance(refined_df)

        if dominance_results and score < best_score:
            best_score = score
            best_config = initial_config.copy()
            no_improvement_count = 0  # Reset the failure counter

            print(f"Iteration {iteration + 1}: New best configuration found!")
            print(f"Score: {score}, Outputs: {dominance_results}")

            # Check if it's a perfect configuration
            if score == 0 and len(dominance_results) == len(input_combinations):
                print("Perfect decoder-like behavior achieved!")
                print("Final Decoder Configuration:")
                print(json.dumps(best_config, indent=4))

                print("Outputs for each input state:")
                for input_state in input_combinations:
                    print(f"InputState: {input_state}")
                    for _, row in refined_df[
                        (refined_df['InputState_Heater36'] == input_state[0]) &
                        (refined_df['InputState_Heater37'] == input_state[1])
                    ].iterrows():
                        print({
                            'Output1': row['Output1'],
                            'Output2': row['Output2'],
                            'Output3': row['Output3'],
                            'Output4': row['Output4']
                        })
                break
        else:
            print("Conflict or no improvement, reverting to the previous best configuration...")
            initial_config = best_config.copy()
            no_improvement_count += 1

        # Stop refinement if no improvement after 50 iterations
        if no_improvement_count >= 50:
            print("No improvement after 50 iterations. Stopping refinement.")
            break

    print(f"Refinement complete. Best Score: {best_score}")
    print(f"Best Configuration: {best_config}")
