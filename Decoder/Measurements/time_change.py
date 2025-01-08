import datetime
import serial
import time
import pyvisa
import json
import keyboard
import os
import csv
# This script doesnt work as it should, but by changing the trigger one can get the image on the scope
# and then take a screenshot

# Serial port configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600

def configure_trigger(scope, trigger_channel='CHANnel1', trigger_level=2.0):
    """
    Configure scope with single-shot trigger settings for voltage step capture
    """
    # Stop acquisition first
    scope.write(":STOP")
    
    # Configure trigger settings
    scope.write(":TRIGger:MODE EDGE")
    scope.write(f":TRIGger:EDGE:SOURce {trigger_channel}")
    scope.write(f":TRIGger:EDGE:LEVel {trigger_level}")  # Trigger at 2V
    scope.write(":TRIGger:EDGE:SLOPe POSitive")  # Trigger on rising edge
    scope.write(":TRIGger:SWEep SINGLE")  # Set to single trigger mode
    
    # Verify trigger settings
    source = scope.query(":TRIGger:EDGE:SOURce?").strip()
    level = scope.query(":TRIGger:EDGE:LEVel?").strip()
    print(f"Trigger configured - Source: {source}, Level: {level}V")
    
    # Start acquisition
    scope.write(":RUN")

def init_hardware():
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    if not resources:
        raise Exception("No VISA resources found. Make sure the oscilloscope is connected.")
    scope = rm.open_resource(resources[0])
    scope.timeout = 5000

    # Initialize channels
    channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
    for channel in channels:
        scope.write(f':{channel}:DISPlay ON')
        scope.write(f':{channel}:SCALe 2')
        scope.write(f':{channel}:OFFSet -6')

    # Set memory depth to maximum available
    scope.write(":ACQuire:MDEPth AUTO")  # Use maximum available memory
    
    # Set timebase for fast edge capture (100µs/div for edge detail)
    scope.write(":TIMebase:SCALe 0.00002")  # 100µs per division = 1ms total window
    scope.write(":TIMebase:POSition -0.00005")  # Show 200µs of pre-trigger data
    
    # Set trigger mode to single
    scope.write(":TRIGger:SWEep SINGLE")
    
    # Initialize serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    
    return scope, ser

def send_heater_values(ser, heater_values):
    voltage_message = "".join(f"{heater},{value};" for heater, value in heater_values.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()

def save_waveform(scope, channels, filename):
    try:
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Channel", "Time (s)", "Voltage (V)"])
            
            for channel in channels:
                # Set source channel
                scope.write(f":WAVeform:SOURce {channel}")
                scope.write(":WAVeform:FORMat ASCii")
                
                # Get memory depth info
                actual_points = int(scope.query(":WAVeform:POINts?"))
                print(f"Number of points for {channel}: {actual_points}")
                
                # Get waveform data
                data = scope.query(":WAVeform:DATA?")
                
                # Get time parameters
                time_step = float(scope.query(":WAVeform:XINCrement?"))
                time_offset = float(scope.query(":WAVeform:XORigin?"))
                time_reference = float(scope.query(":WAVeform:XREFerence?"))
                
                # Convert data to voltages
                voltages = [float(v) for v in data.split(",")]
                
                # Generate time values centered around trigger
                times = [(i - time_reference) * time_step + time_offset 
                        for i in range(len(voltages))]
                
                # Write data to CSV
                for t, v in zip(times, voltages):
                    # Save points within ±500µs window to see edge detail
                    if -0.0005 <= t <= 0.0005:
                        csv_writer.writerow([channel, t, v])
                        
        print(f"Waveform saved: {filename}")
        print(f"Capture window: {min(times):.6f}s to {max(times):.6f}s")
        
    except Exception as e:
        print(f"Error saving waveform: {e}")


def set_and_capture(scope, ser, heater_values, input_a, input_b, channels, save_dir=r"C:\Users\noelp\Documents\Kanazawa\captures"):
    try:
        # Set new heater values
        heater_values[36], heater_values[37] = input_a, input_b
        send_heater_values(ser, heater_values)
        
        # Force trigger to ready state
        scope.write(":TFORce")
        # Wait for the change to occur
        time.sleep(1.5)  
        
        # Save the data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_waveform(scope, channels, os.path.join(save_dir, f"waveform_{timestamp}.csv"))
        
        print(f"Captured data for Input A: {input_a}, Input B: {input_b}")
        
    except Exception as e:
        print(f"Error during set and capture: {e}")

def print_help():
    print("\nControls:")
    print("1: Set Input A,B to 0,0 (0.1V, 0.1V)")
    print("2: Set Input A,B to 0,1 (0.1V, 4.9V)")
    print("3: Set Input A,B to 1,0 (4.9V, 0.1V)")
    print("4: Set Input A,B to 1,1 (4.9V, 4.9V)")
    print("q: Quit")

def main():
    try:
        initial_config = {
            "0": 0.10, "1": 0.10, "2": 0.10, "3": 0.10, "4": 4.90, "5": 0.10, "6": 0.10,
            "7": 4.90, "8": 4.90, "9": 4.90, "10": 4.90, "11": 3.70, "12": 0.10, "13": 0.10,
            "14": 4.90, "15": 4.90, "16": 3.70, "17": 0.10, "18": 3.70, "19": 1.00, "20": 0.10,
            "21": 0.10, "22": 1.00, "23": 4.90, "24": 3.70, "25": 0.10, "26": 4.90, "27": 4.90,
            "28": 4.90, "29": 4.90, "30": 3.70, "31": 2.50, "32": 3.70, "33": 0.01, "34": 0.10,
            "35": 0.50, "36": 0.10, "37": 0.01, "38": 0.01, "39": 0.01
        }
        
        heater_values = {int(k): float(v) for k, v in initial_config.items()}
        scope, ser = init_hardware()
        
        print_help()
        channels = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
        
        # Configure trigger for channel 1
        configure_trigger(scope, 'CHANnel2', trigger_level=4)
        
        # Force trigger to ready state
        scope.write(":TFORce")

        while True:
            if keyboard.is_pressed('1'):
                set_and_capture(scope, ser, heater_values, 0.1, 0.1, channels)
                time.sleep(0.5)
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
        if 'scope' in locals():
            scope.write(":RUN")
            scope.close()
        if 'ser' in locals():
            ser.close()
        print("\nProgram terminated.")

if __name__ == "__main__":
    main()