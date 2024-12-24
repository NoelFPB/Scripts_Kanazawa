import serial
import time
import pyvisa
import csv
import random
import json  # To save heater configurations as strings

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

# Heater values for all 35 heaters
heater_values = {i: 0.0 for i in range(35)}  # Initialize all heaters to 0.0V

# Fixed heaters in Layer 1 (non-input heaters)
fixed_layer1_heaters = [0, 1, 2, 5, 6]
for heater in fixed_layer1_heaters:
    heater_values[heater] = 0.01  # Fixed at 0.01V

# Input configurations for heaters 3 and 4 (Layer 1 middle heaters)
input_combinations = [(0.1, 0.1), (0.1, 4.9), (4.9, 0.1), (4.9, 4.9)]

# Modifiable heaters
modifiable_heaters = list(range(7, 35))

# Voltage range for modifiable heaters
voltage_range = [0.1, 2.5, 4.9]  # Possible random voltages

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

# Main function to test random configurations
def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)

    results = []  # Store results

    # Test multiple random heater configurations
    for config_num in range(100):  # Test 100 random configurations
        print(f"Testing Heater Configuration {config_num + 1}...")

        # Assign random voltages to modifiable heaters
        for heater in modifiable_heaters:
            heater_values[heater] = random.choice(voltage_range)

        # Iterate over all InputStates
        for input_state in input_combinations:
            # Apply InputState to heaters 3 and 4
            heater_values[3], heater_values[4] = input_state

            # Send heater values
            send_heater_values(ser)

            print(f"Voltage sent. Measuring outputs for InputState {input_state}...")
            outputs = measure_outputs()
            if outputs == [None, None, None, None]:
                print("Measurement failed. Skipping...")
                continue

            # Save results, including heater configuration
            heater_config_json = json.dumps(heater_values)  # Convert heater configuration to JSON
            results.append([config_num + 1, input_state, outputs, heater_config_json])
            print(f"Config {config_num + 1}, InputState {input_state} -> Outputs: {outputs}")

    ser.close()
    print("All tests completed. Saving results to CSV...")

    # Save results to CSV
    with open('decoder_test_results_with_configurations.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ConfigNumber", "InputState_Heater3", "InputState_Heater4", 
                         "Output1", "Output2", "Output3", "Output4", "HeaterConfiguration"])
        for result in results:
            writer.writerow([result[0], result[1][0], result[1][1], 
                             result[2][0], result[2][1], result[2][2], result[2][3], result[3]])

    print("Results saved to 'decoder_test_results_with_configurations.csv'.")

if __name__ == "__main__":
    main()
