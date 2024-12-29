import serial
import time
import pyvisa
import random
import json
import pandas as pd
import csv


# Flag to toggle between preset values and random initialization
use_preset = False  # Set to True to use preset values, False for random initialization

# Preset heater values (example values, replace as needed)
preset_heater_values = {"0": 4.9, "1": 0.1, "2": 2.0, "3": 1.0, "4": 4.0, "5": 2.0, "6": 3.0, "7": 4.0, "8": 0.1, "9": 3.0, "10": 0.1, "11": 1.0, "12": 4.0, "13": 0.1, "14": 1.0, "15": 3.0, "16": 0.1, "17": 3.0, "18": 3.0, "19": 4.0, "20": 0.1, "21": 1.0, "22": 0.1, "23": 0.1, "24": 4.9, "25": 0.1, "26": 2.0, "27": 0.1, "28": 3.0, "29": 1.0, "30": 0.1, "31": 0.1, "32": 3.0, "33": 0.01, "34": 0.01, "35": 0.01, "36": 4.9, "37": 4.9, "38": 0.01, "39": 0.01}

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

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

# Function to initialize heater values
def initialize_heater_values():
    global heater_values
    if use_preset:
        # Use preset values only for modifiable heaters
        for heater, value in preset_heater_values.items():
            heater = int(heater)  # Ensure the heater key is an integer
            if heater in modifiable_heaters:  # Only update modifiable heaters
                heater_values[heater] = value
        print("Using preset heater values.")
    else:
        # Initialize only modifiable heaters with random values
        for heater in modifiable_heaters:
            heater_values[heater] = random.choice(voltage_range)
        print("Using random heater values.")

    print(heater_values)  # Print the initial heater configuration for debugging

# Function to send heater values
def send_heater_values(ser):
    voltage_message = "".join(f"{heater},{value};" for heater, value in heater_values.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.4) # Delay so that the values can be stable

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

def calculate_decoder_score(df, dominant_outputs):
    """
    Calculate a score that prioritizes increasing the difference for the dominant output.
    """
    total_score = 0

    for idx, row in df.iterrows():
        outputs = [row['Output1'], row['Output2'], row['Output3'], row['Output4']]
        dominant_index = dominant_outputs[idx]
        dominant_value = outputs[dominant_index]

        # Remove the dominant output and find the next highest
        other_outputs = outputs[:]
        other_outputs.pop(dominant_index)
        next_highest = max(other_outputs)

        # Reward the difference
        total_score += dominant_value - next_highest

    # Normalize by the number of input states
    return total_score / len(df)



def calculate_gradient(dominant_outputs, heater, config, refined_df, learning_rate=0.5):
    """
    Calculate the gradient for a specific heater by slightly adjusting its value
    and observing the impact on the decoder score (which rewards higher differences).
    """
    original_value = config[heater]

    # Calculate score for a small positive adjustment
    config[heater] = max(0.1, min(4.9, original_value + 0.5))  # Adjust heater upwards
    positive_df = test_configuration(config, refined_df)  # Test updated configuration
    positive_score = calculate_decoder_score(positive_df, dominant_outputs)  # Evaluate score

    # Calculate score for a small negative adjustment
    config[heater] = max(0.1, min(4.9, original_value - 0.5))  # Adjust heater downwards
    negative_df = test_configuration(config, refined_df)  # Test updated configuration
    negative_score = calculate_decoder_score(negative_df, dominant_outputs)  # Evaluate score

    # Restore the original value of the heater
    config[heater] = original_value

    # Calculate gradient as the rate of change of the score with respect to the heater
    gradient = (positive_score - negative_score) / 0.2  # Use the step size (0.2 = 2 * 0.1 adjustment)

    # Adjust the heater value in the direction that improves the score
    adjusted_value = original_value - learning_rate * gradient

    # Ensure the adjusted value remains within the bounds
    return max(0.1, min(4.9, adjusted_value))


def test_configuration(config):
    """
    Test the configuration for all input combinations and return the updated DataFrame.
    """
    iteration_results = []

    for input_state in input_combinations:
        heater_values[36], heater_values[37] = input_state
        send_heater_values(ser)
        
        outputs = measure_outputs()
        
        if outputs == [None, None, None, None]:
            continue

        iteration_results.append({
            'InputState_Heater36': input_state[0],
            'InputState_Heater37': input_state[1],
            'Output1': outputs[0],
            'Output2': outputs[1],
            'Output3': outputs[2],
            'Output4': outputs[3],
            'HeaterConfiguration': json.dumps(config)
        })

        # Print only inputs and outputs from iteration_results
        for result in iteration_results:
            print("Inside test configuration")
            print(f"Input States: Heater36={result['InputState_Heater36']}, Heater37={result['InputState_Heater37']}")
            print(f"Outputs: Output1={result['Output1']}, Output2={result['Output2']}, Output3={result['Output3']}, Output4={result['Output4']}")
            print()

    return pd.DataFrame(iteration_results)

def determine_dominant_outputs(df):
    """
    Identify the dominant output for each combination of inputs.
    Returns a dictionary mapping input index to the dominant output index.
    """
    dominant_outputs = {}
    for idx, row in df.iterrows():
        outputs = [row['Output1'], row['Output2'], row['Output3'], row['Output4']]
        dominant_index = outputs.index(max(outputs))  # Index of the highest output
        dominant_outputs[idx] = dominant_index
    return dominant_outputs

# Main function
def main():
    # Initialize heater values based on the flag
    initialize_heater_values()

    print("Searching for a Decoder-Like Configuration...")
    refinement_iterations = 400  # Number of refinement iterations
    decoder_found = False

    initial_config = None  # Initialize initial_config outside the loop
    while not decoder_found:
        # Randomly set a configuration if not using preset
        if not use_preset:
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
            print(initial_config)
            # Validate and complete initial_config
            # for heater in modifiable_heaters:
            #     if heater not in initial_config:
            #         initial_config[heater] = 0.1  # Default to the minimum voltage
            # break

    if initial_config is None:
        print("No valid decoder-like configuration found. Exiting...")
        ser.close()
        return


    # Define the log file path
    log_file = "optimization_log_with_outputs.csv"

    # Initialize the log file with headers
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Add columns for iteration details and outputs
        writer.writerow(["Iteration", "Score", "Best Score", "Configuration", "Outputs"])

    best_score = float('-inf')  # Start with a very low score for maximization
    score_history = []

    # Determine dominant outputs based on the initial configuration
    dominant_outputs = determine_dominant_outputs(test_configuration(initial_config))

    best_score = float('-inf')
    for iteration in range(refinement_iterations):
        print(f"Iteration {iteration + 1}...")

        # Save previous configuration for recovery if needed
        previous_config = initial_config.copy()

        # Apply gradient improvement for each heater
        for heater in modifiable_heaters:
            initial_config[heater] = calculate_gradient(dominant_outputs, heater, initial_config, df)

        # Test the updated configuration
        refined_df = test_configuration(initial_config, df)
        score = calculate_decoder_score(refined_df, dominant_outputs)
        print(f"Iteration {iteration + 1}, Score: {score}")

        # Validate that the dominant outputs remain unchanged
        new_dominant_outputs = determine_dominant_outputs(refined_df)
        if new_dominant_outputs != dominant_outputs:
            print("Dominant outputs changed. Reverting to previous configuration.")
            initial_config = previous_config.copy()
            continue

        # Update best score and configuration if improved
        if score > best_score:
            best_score = score
            best_config = initial_config.copy()
            print(f"Improved configuration! Score: {score}")


        # Log the details of the current iteration
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration + 1, score, best_score, initial_config.copy(), outputs])

        # Optional: Early stop condition if improvement is above a dynamic threshold
        if len(score_history) > 5 and all(
            abs(score_history[-i] - score_history[-i - 1]) < 0.01 for i in range(1, 5)
        ):
            print("Convergence plateau detected; stopping optimization.")
            break

    print(f"Refinement complete. Best Score: {best_score}")
    print(f"Best Configuration: {best_config}")


if __name__ == "__main__":
    main()