import serial
import time
import pyvisa
import keyboard
import os


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
            initial_config ={0: 0.6, 1: 2.5, 2: 2.2, 3: 4.9, 4: 1.7, 5: 1.9, 6: 2.9, 7: 2.8, 8: 0.5, 9: 4.5, 10: 3.8, 11: 2.4, 12: 1.2, 13: 0.5, 14: 3.5, 15: 2.5, 16: 0.9, 17: 4.4, 18: 3.2, 19: 4.0, 20: 3.4, 21: 4.0, 22: 2.4, 23: 3.0, 24: 4.4, 25: 4.3, 26: 0.5, 27: 3.5, 28: 2.8, 29: 4.5, 30: 2.0, 31: 3.1, 32: 0.8, 33: 0.01, 34: 0.01, 35: 0.01, 36: 0.01, 37: 0.0, 38: 0.0, 39: 0.01}

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
                    
                    #clear_screen()
                    #print_help()
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