import serial
import time
import pyvisa
import keyboard
import os
import numpy as np


# USING as input 36 and 37
# Serial port configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Define heater combinations for each key
HEATER_COMBINATIONS = {
    '1': (0.1, 0.1),  # 00
    '2': (0.1, 4.9),  # 01
    '3': (4.9, 0.1),  # 10
    '4': (4.9, 4.9)   # 11
}

# Measurement parameters
NUM_MEASUREMENTS = 10  # Number of measurements to average
MEASUREMENT_DELAY = 0.1  # Delay between measurements in seconds

# Channel selection - set to True for channels you want to measure
# Channel indices: 0=CH1, 1=CH2, 2=CH3, 3=CH4
CHANNELS_TO_MEASURE = [0, True, 0, 0]  # By default, measure all channels

def init_hardware():
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

    # Initialize serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)  # Wait for connection to stabilize
    
    return scope, ser

def measure_outputs(scope):
    try:
        # Initialize arrays to store the measurements
        measurements = [[] for _ in range(4)]
        channel_names = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
        
        # Take multiple measurements for each active channel
        for i in range(NUM_MEASUREMENTS):
            try:
                # Measure each active channel
                for ch_idx, is_active in enumerate(CHANNELS_TO_MEASURE):
                    if is_active:
                        value = float(scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,{channel_names[ch_idx]}'))
                        measurements[ch_idx].append(value)
                
                # Wait a bit between measurements to ensure independent readings
                time.sleep(MEASUREMENT_DELAY)
                
            except Exception as e:
                print(f"Error in measurement {i+1}: {e}")
        
        # Calculate averages and round to 5 decimal places
        averages = [None] * 4  # Initialize with None values
        for ch_idx, channel_data in enumerate(measurements):
            if not CHANNELS_TO_MEASURE[ch_idx] or not channel_data:
                continue  # Skip inactive channels
                
            # Calculate average and round to 5 decimal places
            average = round(sum(channel_data) / len(channel_data), 5)
            averages[ch_idx] = average
        
        return averages
    except Exception as e:
        print(f"Error measuring outputs: {e}")
        return [None, None, None, None]

def send_heater_values(ser, heater_values):
    voltage_message = "".join(f"{heater},{value};" for heater, value in heater_values.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.01)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_help():
    print("\nControls:")
    print("1: Set Input A,B to 0,0 (0.1V, 0.1V)")
    print("2: Set Input A,B to 0,1 (0.1V, 4.9V)")
    print("3: Set Input A,B to 1,0 (4.9V, 0.1V)")
    print("4: Set Input A,B to 1,1 (4.9V, 4.9V)")
    print("d: Display all heater values")
    print("q: Quit")

def display_all_heaters(heater_values):
    print("\nAll Heater Values:")
    for i in range(0, 40, 8):
        values = [f"H{j}: {heater_values[j]:.2f}V" for j in range(i, min(i+8, 40))]
        print("  ".join(values))
    print("\n")

# The configure_channels function is removed

def main():
    try:
            initial_config = {0: 4.64, 1: 4.62, 2: 2.86, 3: 2.47, 4: 4.49, 5: 4.7, 6: 2.56, 7: 3.66, 8: 1.43, 9: 4.39, 10: 3.65, 11: 3.9, 12: 3.01, 13: 4.7, 14: 3.82, 15: 0.77, 16: 0.58, 17: 0.69, 18: 2.79, 19: 2.78, 20: 3.59, 21: 3.2, 22: 2.27, 23: 3.13, 24: 3.04, 25: 2.2, 26: 3.46, 27: 4.41, 28: 4.8, 29: 2.31, 30: 1.28, 31: 2.42, 32: 4.9, 33: 0.01, 34: 0.01, 35: 0.01, 36: 0.0, 37: 0.0, 38: 0.01, 39: 0.01}
            heater_values = {int(k): float(v) for k, v in initial_config.items()}
            scope, ser = init_hardware()

            
            print_help()
            display_all_heaters(heater_values)
            
            prev_value_36 = heater_values[36]
            prev_value_37 = heater_values[37]
            
            while True:
                value_changed = False
                for key in ['1', '2', '3', '4']:
                    if keyboard.is_pressed(key):
                        heater_values[36], heater_values[37] = HEATER_COMBINATIONS[key]
                        value_changed = True
                        time.sleep(0.2)
                        break
                
                if keyboard.is_pressed('d'):
                    clear_screen()
                    print_help()
                    display_all_heaters(heater_values)
                    time.sleep(0.2)

                elif keyboard.is_pressed('q'):
                    break
                    
                if value_changed:
                    send_heater_values(ser, heater_values)
                    time.sleep(0.2)
                    # Perform multiple measurements and get the average
                    outputs = measure_outputs(scope)
                    
                    # Log the change
                    print(f"\nChanged inputs:")
                    print(f"Input A (H36): {prev_value_36:.2f}V -> {heater_values[36]:.2f}V")
                    print(f"Input B (H37): {prev_value_37:.2f}V -> {heater_values[37]:.2f}V")
                    
                    print(f"\nAverage outputs (from {NUM_MEASUREMENTS} measurements):")
                    # Only show active channels
                    for ch_idx, is_active in enumerate(CHANNELS_TO_MEASURE):
                        if is_active and outputs[ch_idx] is not None:
                            print(f"O{ch_idx+1}: {outputs[ch_idx]}V")
                    
                    prev_value_36 = heater_values[36]
                    prev_value_37 = heater_values[37]
                
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        ser.close()
        scope.close()
        print("\nProgram terminated.")

if __name__ == "__main__":
    main()