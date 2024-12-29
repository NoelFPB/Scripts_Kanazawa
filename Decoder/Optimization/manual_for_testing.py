import serial
import time
import pyvisa
import json
import keyboard
import os
import csv

# In this one I can load a configuration of heaters and test it manually.

# Serial port configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600

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
        output1 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel1')), 5)
        output2 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel2')), 5)
        output3 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel3')), 5)
        output4 = round(float(scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel4')), 5)
        return [output1, output2, output3, output4]
    except Exception as e:
        print(f"Error measuring outputs: {e}")
        return [None, None, None, None]

def send_heater_values(ser, heater_values):
    voltage_message = "".join(f"{heater},{value};" for heater, value in heater_values.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(2)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_help():
    print("\nControls:")
    print("1: Set Input A (Heater 36) to 0.1V")
    print("2: Set Input A (Heater 36) to 4.9V")
    print("3: Set Input B (Heater 37) to 0.1V")
    print("4: Set Input B (Heater 37) to 4.9V")
    print("d: Display all heater values")
    print("q: Quit")

def display_all_heaters(heater_values):
    print("\nAll Heater Values:")
    for i in range(0, 40, 8):
        values = [f"H{j}: {heater_values[j]:.2f}V" for j in range(i, min(i+8, 40))]
        print("  ".join(values))
    print("\n")

def main():
    try:
        # Create log file
        with open('heater_changes.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow([
                'Input_A_Old', 'Input_A_New',
                'Input_B_Old', 'Input_B_New',
                'Output1', 'Output2', 'Output3', 'Output4'
            ])
            
            # This combination works, and it was found with the hill
            # Load initial heater configuration
            initial_config = {"0": 0.10, "1": 2.50, "2": 4.90, "3": 2.50, "4": 0.10, "5": 2.50, "6": 2.50,
                            "7": 2.50, "8": 2.50, "9": 4.90, "10": 2.50, "11": 4.90, "12": 2.50, "13": 4.90,
                            "14": 2.50, "15": 2.50, "16": 4.90, "17": 0.10, "18": 0.10, "19": 0.10, "20": 0.10,
                            "21": 4.90, "22": 0.10, "23": 0.10, "24": 0.10, "25": 4.90, "26": 0.10, "27": 4.90,
                            "28": 2.50, "29": 0.10, "30": 4.90, "31": 4.90, "32": 4.90, "33": 0.01, "34": 0.01,
                            "35": 0.01, "36": 0.01, "37": 0.01, "38": 0.01, "39": 0.01
                            }
            
            heater_values = {int(k): float(v) for k, v in initial_config.items()}
            scope, ser = init_hardware()
            
            print_help()
            display_all_heaters(heater_values)
            
            prev_value_36 = heater_values[36]
            prev_value_37 = heater_values[37]
            
            while True:
                value_changed = False
                input_a_old = prev_value_36
                input_b_old = prev_value_37
                
                if keyboard.is_pressed('1'):
                    heater_values[36] = 0.1
                    value_changed = True
                    time.sleep(0.2)
                elif keyboard.is_pressed('2'):
                    heater_values[36] = 4.9
                    value_changed = True
                    time.sleep(0.2)
                elif keyboard.is_pressed('3'):
                    heater_values[37] = 0.1
                    value_changed = True
                    time.sleep(0.2)
                elif keyboard.is_pressed('4'):
                    heater_values[37] = 4.9
                    value_changed = True
                    time.sleep(0.2)
                elif keyboard.is_pressed('d'):
                    clear_screen()
                    print_help()
                    display_all_heaters(heater_values)
                    time.sleep(0.2)
                elif keyboard.is_pressed('q'):
                    break
                    
                if value_changed:
                    send_heater_values(ser, heater_values)
                    outputs = measure_outputs(scope)
                    
                    # Log the change
                    csv_writer.writerow([
                        input_a_old, heater_values[36],
                        input_b_old, heater_values[37],
                        outputs[0], outputs[1], outputs[2], outputs[3]
                    ])
                    csvfile.flush()
                    
                    clear_screen()
                    print_help()
                    print(f"\nChanged inputs:")
                    if prev_value_36 != heater_values[36]:
                        print(f"Input A (H36): {prev_value_36:.2f}V -> {heater_values[36]:.2f}V")
                    if prev_value_37 != heater_values[37]:
                        print(f"Input B (H37): {prev_value_37:.2f}V -> {heater_values[37]:.2f}V")
                    
                    print(f"\nCurrent outputs:")
                    print(f"O1: {outputs[0]}V")
                    print(f"O2: {outputs[1]}V")
                    print(f"O3: {outputs[2]}V")
                    print(f"O4: {outputs[3]}V")
                    
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