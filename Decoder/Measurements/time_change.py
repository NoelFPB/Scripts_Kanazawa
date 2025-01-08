import datetime
import serial
import time
import pyvisa
import json
import keyboard
import os
import csv

# Serial port configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600

# Define heater combinations for each key
HEATER_COMBINATIONS = {
    '1': (0.1, 0.1),  # 00
    '2': (0.1, 4.9),  # 01
    '3': (4.9, 0.1),  # 10
    '4': (4.9, 4.9)   # 11
}

def configure_trigger(scope, trigger_channel='CHANnel1', trigger_level=2.5):
    # Configure trigger settings
    scope.write(f":TRIGger:EDGE:SOURce {trigger_channel}")
    scope.write(":TRIGger:EDGE:LEVel {trigger_level}")
    scope.write(":TRIGger:EDGE:SLOPe POSitive")
    scope.write(":TRIGger:MODE NORMal")  # Normal mode triggers only on events
    scope.write(":TIMebase:POSition 50")  # Set trigger position to 50% of screen

def set_timebase(scope, time_per_div):
    # Set timebase (horizontal scale)
    scope.write(f":TIMebase:SCALe {time_per_div}")

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
    time.sleep(2.5)

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

def save_screenshot(scope, save_dir="captures"):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"screenshot_{timestamp}.png")
        scope.write(":HARDcopy:INKSaver OFF")
        scope.write(":DISPlay:DATA?")
        raw_data = scope.read_raw()
        with open(filename, 'wb') as f:
            f.write(raw_data)
        print(f"Screenshot saved: {filename}")
    except Exception as e:
        print(f"Error saving screenshot: {e}")

def save_waveform(scope, channels, save_dir="waveforms"):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"waveform_{timestamp}.csv")
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Channel", "Time (s)", "Voltage (V)"])
            for channel in channels:
                scope.write(f":WAVeform:SOURce {channel}")
                scope.write(":WAVeform:FORMat ASCii")
                data = scope.query(":WAVeform:DATA?")
                time_step = float(scope.query(":WAVeform:XINCrement?"))
                time_offset = float(scope.query(":WAVeform:XORigin?"))
                voltages = [float(v) for v in data.split(",")]
                times = [time_offset + i * time_step for i in range(len(voltages))]
                for t, v in zip(times, voltages):
                    csv_writer.writerow([channel, t, v])
        print(f"Waveform saved: {filename}")
    except Exception as e:
        print(f"Error saving waveform: {e}")

def set_and_capture(scope, ser, heater_values, input_a, input_b, channels, save_dir="captures"):
    try:
        # Set new heater values
        heater_values[36], heater_values[37] = input_a, input_b
        send_heater_values(ser, heater_values)

        # Wait for the circuit to stabilize
        time.sleep(2)  # Adjust delay if needed

        # Measure and save the waveform
        save_waveform(scope, channels, save_dir)
        save_screenshot(scope, save_dir)

        print(f"Captured waveform for Input A: {input_a}, Input B: {input_b}")
    except Exception as e:
        print(f"Error during set and capture: {e}")


def main():
    try:
        scope, ser = init_hardware()
        print_help()
        channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
        heater_values = {36: 0.1, 37: 0.1}  # Default heater values
        configure_trigger(scope, 'CHANnel1')
        configure_trigger(scope, 'CHANnel2')
        configure_trigger(scope, 'CHANnel3')
        configure_trigger(scope, 'CHANnel4')
        set_timebase(scope, 0.01)
        while True:
            # Detect key presses and capture changes
            if keyboard.is_pressed('1'):
                set_and_capture(scope, ser, heater_values, 0.1, 0.1, channels)
                time.sleep(0.5)  # Debounce
            elif keyboard.is_pressed('2'):
                set_and_capture(scope, ser, heater_values, 0.1, 4.9, channels)
                time.sleep(0.5)
            elif keyboard.is_pressed('3'):
                set_and_capture(scope, ser, heater_values, 4.9, 0.1, channels)
                time.sleep(0.5)
            elif keyboard.is_pressed('4'):
                set_and_capture(scope, ser, heater_values, 4.9, 4.9, channels)
                time.sleep(0.5)
            elif keyboard.is_pressed('q'):
                break

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        ser.close()
        scope.close()
        print("\nProgram terminated.")

if __name__ == "__main__":
    main()

