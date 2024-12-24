import serial
import time
import pyvisa
import itertools
import random

# Serial port configuration
SERIAL_PORT = 'COM4'  # Replace with your serial port
BAUD_RATE = 9600

# Initialize oscilloscope
rm = pyvisa.ResourceManager()
resources = rm.list_resources()
if not resources:
    raise Exception("No VISA resources found. Make sure the oscilloscope is connected.")
scope = rm.open_resource(resources[0])
scope.timeout = 5000  # Set timeout to 5 seconds

# Initial
channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
for channel in channels:
    scope.write(f':{channel}:DISPlay ON')  # Activate channels
    scope.write(f':{channel}:SCALe 2')    # Set scale
    scope.write(f':{channel}:OFFSet -6')  # Set offset


# Heater values grouped by layers
heater_values = {i: 0.0 for i in range(35)}  # 35 heaters in total
layers = {
    0: list(range(0, 7)),  # Layer 1: Heaters 0-6
    1: list(range(7, 14)),  # Layer 2: Heaters 7-13
    2: list(range(14, 21)),  # Layer 3: Heaters 14-20
    3: list(range(21, 28)),  # Layer 4: Heaters 21-27
    4: list(range(28, 35)),  # Layer 5: Heaters 28-34
}

# Voltage range
voltage_range = [0.1, 2.5, 4.9]  # Low, mid, high voltages
num_combinations_per_layer_pair = 100  # Limit random samples per layer pair

# Input configurations for heaters 3 and 4 (Layer 1 middle heaters)
input_combinations = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]

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
        return output1, output2, output3, output4
    except Exception as e:
        print(f"Error measuring outputs: {e}")
        return [None, None, None, None]

# Main loop
def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)

    results = []  # Store all results in memory

    # Iterate over all input combinations for Layer 1
    for input_state in input_combinations:
        print(f"Testing input state: {input_state}...")

        # Set fixed heaters in Layer 1 (because I am using 3 and 4 as the bit inputs)
        for heater in [0, 1, 2, 5, 6]:
            heater_values[heater] = 0.01

        # Apply input state to heaters 3 and 4
        heater_values[3], heater_values[4] = input_state

        # Test combinations for multiple layers simultaneously
        # Exclude Layer 0 from the combinations
        layer_combinations = list(itertools.combinations(range(1, len(layers)), 2))  # Start from Layer 1

        for layer_pair in layer_combinations:
            print(f"Testing Layer Pair: {layer_pair} with input state {input_state}...")

            # Get heaters for the selected layers
            heaters_in_pair = [heater for layer in layer_pair for heater in layers[layer]]

            # Generate combinations for the selected layers
            all_combinations = list(itertools.product(voltage_range, repeat=len(heaters_in_pair)))
            sampled_combinations = random.sample(all_combinations, min(num_combinations_per_layer_pair, len(all_combinations)))

            for combination in sampled_combinations:
                # Reset all other heaters to baseline
                for key in heater_values.keys():
                    if key not in layers[0]:  # Exclude Layer 1 heaters
                        heater_values[key] = 0.0

                # Apply the combination to the selected layers
                for heater, voltage in zip(heaters_in_pair, combination):
                    heater_values[heater] = voltage

                print(f"Testing combination {combination} for Layer Pair {layer_pair} with input state {input_state}...")
                send_heater_values(ser)

                print("Voltage sent. Measuring outputs...")
                outputs = measure_outputs()
                if outputs == [None, None, None, None]:
                    print("Measurement failed. Skipping...")
                    break

                # Save results in memory
                results.append([input_state, layer_pair, combination] + list(outputs))
                print(f"Input {input_state}, Layer Pair {layer_pair}, Combination {combination} -> Outputs: {outputs}")

    ser.close()
    print("All cases tested. Saving results to CSV...")

    # Save all results to CSV at the end
    with open('multi_layer_test_with_decoder.csv', 'w', newline='') as csvfile:
        import csv
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["InputState", "LayerPair", "Combination", "Output1", "Output2", "Output3", "Output4"])
        csvwriter.writerows(results)

    print("Results saved to 'multi_layer_test_with_decoder.csv'.")

if __name__ == "__main__":
    main()
