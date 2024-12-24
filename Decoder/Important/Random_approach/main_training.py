import serial
import time
import pyvisa
import random
import json
import pandas as pd

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
voltage_range = [0.1, 2.5, 3.3, 4.9]

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
    for _, row in df.iterrows():
        outputs = [row['Output1'], row['Output2'], row['Output3'], row['Output4']]
        max_output = max(outputs)
        distinct_count = sum(1 for o in outputs if max_output - o >= 1.0)
        if distinct_count != 3:
            return False
    return True

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

    all_results = []
    decoder_like_results = {}

    for config_num in range(100):
        print(f"Testing Heater Configuration {config_num + 1}...")

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
                'ConfigNumber': config_num + 1,
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
        all_results.extend(config_results)

        if check_decoder_behavior(df):
            decoder_like_results[config_num + 1] = config_results

    ser.close()

    print("Refining Decoder-Like Configurations...")
    refinement_iterations = 50  # Number of iterations to refine each configuration
    refined_results = []

    for config, results in decoder_like_results.items():
        print(f"Refining ConfigNumber {config}...")
        initial_config = json.loads(results[0]['HeaterConfiguration'])
        best_score = float('inf')  # Initialize with a high score
        best_config = initial_config.copy()

        for iteration in range(refinement_iterations):
            print(f"Iteration {iteration + 1} for ConfigNumber {config}...")

            # Adjust modifiable heaters slightly
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
                    'ConfigNumber': config,
                    'Iteration': iteration + 1,
                    'InputState_Heater36': input_state[0],
                    'InputState_Heater37': input_state[1],
                    'Output1': outputs[0],
                    'Output2': outputs[1],
                    'Output3': outputs[2],
                    'Output4': outputs[3],
                    'HeaterConfiguration': json.dumps(initial_config)
                })

            # Evaluate behavior using the metric
            refined_df = pd.DataFrame(iteration_results)
            score = calculate_decoder_score(refined_df)
            print(f"Iteration {iteration + 1}, Score: {score}")

            # Save the best configuration
            if score < best_score:
                best_score = score
                best_config = initial_config.copy()

            refined_results.extend(iteration_results)

        print(f"Best score for ConfigNumber {config}: {best_score}")
        print(f"Best configuration: {best_config}")

    # Optionally save all results to a CSV
    pd.DataFrame(all_results).to_csv('complete_results_with_configurations.csv', index=False)
    pd.DataFrame(refined_results).to_csv('refined_decoder_configurations.csv', index=False)
    print("All configurations saved to 'complete_results_with_configurations.csv'.")
    print("Refined configurations saved to 'refined_decoder_configurations.csv'.")

if __name__ == "__main__":
    main()
