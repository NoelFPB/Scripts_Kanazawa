import serial
import time
import pyvisa
import keyboard
import os

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

def main():
    try:
        # Create log file
            
            # Load initial heater configuration
            # initial_config = {"0": 0.10, "1": 2.50, "2": 4.90, "3": 2.50, "4": 0.10, "5": 2.50, "6": 2.50,
            #                 "7": 2.50, "8": 2.50, "9": 4.90, "10": 2.50, "11": 4.90, "12": 2.50, "13": 4.90,
            #                 "14": 2.50, "15": 2.50, "16": 4.90, "17": 0.10, "18": 0.10, "19": 0.10, "20": 0.10,
            #                 "21": 4.90, "22": 0.10, "23": 0.10, "24": 0.10, "25": 4.90, "26": 0.10, "27": 4.90,
            #                 "28": 2.50, "29": 0.10, "30": 4.90, "31": 4.90, "32": 4.90, "33": 0.01, "34": 0.01,
            #                 "35": 0.01, "36": 0.01, "37": 0.01, "38": 0.01, "39": 0.01
            #                 }

            # This one works much better than the previous one
            # Still from hill
            # initial_config = {"0": 0.10, "1": 4.90, "2": 4.90, "3": 0.10, "4": 4.90, "5": 0.10, "6": 0.10,
            #      "7": 4.90, "8": 4.90, "9": 0.10, "10": 2.50, "11": 4.90, "12": 4.90, "13": 0.10,
            #      "14": 2.50, "15": 4.90, "16": 4.90, "17": 2.50, "18": 0.10, "19": 0.10, "20": 2.50,
            #      "21": 4.90, "22": 4.90, "23": 0.10, "24": 4.90, "25": 4.90, "26": 0.10, "27": 0.10,
            #      "28": 0.10, "29": 2.50, "30": 2.50, "31": 0.10, "32": 2.50, "33": 0.01, "34": 0.01,
            #      "35": 0.01, "36": 0.01, "37": 0.01, "38": 0.01, "39": 0.01
            #     }
            
            # This one works, but the 2 previous ones stopped working
            # initial_config = {
            #             "0": 0.10, "1": 0.10, "2": 4.90, "3": 2.50, "4": 0.10, "5": 2.50, "6": 2.50, 
            #             "7": 2.50, "8": 4.90, "9": 4.90, "10": 0.10, "11": 0.10, "12": 0.10, "13": 4.90, 
            #             "14": 4.90, "15": 0.10, "16": 0.10, "17": 0.10, "18": 4.90, "19": 0.10, "20": 0.10, 
            #             "21": 2.50, "22": 0.10, "23": 4.90, "24": 0.10, "25": 0.10, "26": 2.50, "27": 0.10, 
            #             "28": 2.50, "29": 4.90, "30": 4.90, "31": 0.10, "32": 0.10, "33": 0.01, "34": 0.01, 
            #             "35": 0.01, "36": 0.01, "37": 0.01, "38": 0.01, "39": 0.01
            #         }
            

            # Best one over all
            # initial_config = {
            #     "0": 0.10, "1": 0.10, "2": 0.10, "3": 0.10, "4": 4.90, "5": 0.10, "6": 0.10,
            #     "7": 4.90, "8": 4.90, "9": 4.90, "10": 4.90, "11": 3.70, "12": 0.10, "13": 0.10,
            #     "14": 4.90, "15": 4.90, "16": 3.70, "17": 0.10, "18": 3.70, "19": 1.00, "20": 0.10,
            #     "21": 0.10, "22": 1.00, "23": 4.90, "24": 3.70, "25": 0.10, "26": 4.90, "27": 4.90,
            #     "28": 4.90, "29": 4.90, "30": 3.70, "31": 2.50, "32": 3.70, "33": 0.01, "34": 0.10,
            #     "35": 0.50, "36": 0.10, "37": 0.01, "38": 0.01, "39": 0.01
            # }


            initial_config = {'0': 4.0, '1': 4.9, '2': 4.5, '3': 4.9, '4': 2.5, '5': 3.0, '6': 3.5, '7': 2.0, '8': 4.9, '9': 2.5, '10': 4.0, '11': 0.1, '12': 4.9, '13': 4.9, '14': 3.0, '15': 1.5, '16': 0.1, '17': 4.5, '18': 2.0, '19': 2.5, '20': 3.0, '21': 0.1, '22': 2.5, '23': 0.1, '24': 2.0, '25': 3.5, '26': 1.5, '27': 2.5, '28': 4.0, '29': 0.1, '30': 0.1, '31': 4.0, '32': 2.5, '33': 0.01, '34': 0.01, '35': 0.5, '36': 0.5, '37': 0.1, '38': 0.1, '39': 0.5}
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
                    outputs = measure_outputs(scope)
                    
                    # Log the change
                    
                    clear_screen()
                    print_help()
                    print(f"\nChanged inputs:")
                    print(f"Input A (H36): {prev_value_36:.2f}V -> {heater_values[36]:.2f}V")
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