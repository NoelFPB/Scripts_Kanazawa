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

def configure_trigger(scope, trigger_channel='CHANnel1', trigger_level=2.5):
    """
    Configure scope with working trigger settings
    """
    # Basic trigger setup without RMT commands
    scope.write(f":TRIGger:EDGE:SOURce {trigger_channel}")
    scope.write(f":TRIGger:EDGE:LEVel {trigger_level}")
    scope.write(":TRIGger:EDGE:SLOPe POSitive")
    scope.write(":TRIGger:MODE SLOPe")


def set_timebase(scope, time_per_div):
    """
    Set timebase with simpler settings
    """
    scope.write(f":TIMebase:SCALe {time_per_div}")

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
        scope.write(f':{channel}:COUPling DC')

    # Set up scope for continuous running
    scope.write(":RUN")
    
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
    #time.sleep(2.5)

def save_waveform(scope, channels, filename):
    try:
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

def capture_screenshot(scope, filename):
    try:
        # Set longer timeout for image transfer
        original_timeout = scope.timeout
        scope.timeout = 30000
        
        # Capture the display data
        scope.write(':DISPLAY:DATA? PNG')
        image_data = scope.read_raw()
        
        # Look for PNG signature
        png_start = image_data.find(b'\x89PNG')
        if png_start >= 0:
            with open(filename, 'wb') as f:
                f.write(image_data[png_start:])
            print(f"Screenshot saved: {filename}")
            return True
        else:
            print("No valid PNG data found in scope response")
            return False
            
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False
    finally:
        scope.timeout = original_timeout   
def start_record(scope):
    """Start the scope's record function"""
    try:
        #scope.write(":RUN")
        scope.write(":RECord:WRECord:ENABle ON")  # Enable record
        scope.write(":RECord:WRECord:OPERate RUN")  # Start recording
        print("Recording started")
    except Exception as e:
        print(f"Error starting record: {e}")

def stop_record(scope):
    """Stop the scope's record function"""
    try:
        
        #scope.write(":RECord:OPERate STOP")  # Disable record
        scope.write(":RECord:WRECord:OPERate STOP")  # Start recording
        
        print("Recording stopped")
    except Exception as e:
        print(f"Error stopping record: {e}")

def set_and_capture(scope, ser, heater_values, input_a, input_b, channels, save_dir=r"C:\Users\noelp\Documents\Kanazawa\captures"):
    try:
 
        # Set new heater values
        heater_values[36], heater_values[37] = input_a, input_b
        send_heater_values(ser, heater_values)
        
        # Wait for the change to occur (around 1.4 seconds)
        time.sleep(1.5)  
        # Start recording
        start_record(scope)
       
        # Stop recording
        #stop_record(scope)
        
        # Save the recorded data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_waveform(scope, channels, os.path.join(save_dir, f"waveform_{timestamp}.csv"))
        capture_screenshot(scope, os.path.join(save_dir, f"screenshot_{timestamp}.png"))
        
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
        
        # Configure basic trigger for each channel
        for channel in channels:
            configure_trigger(scope, channel)
        set_timebase(scope, 0.01)

        # Ensure scope is running
        scope.write(":RUN")

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
        # Ensure scope is running before closing
        if 'scope' in locals():
            scope.write(":RUN")
            scope.close()
        if 'ser' in locals():
            ser.close()
        print("\nProgram terminated.")

if __name__ == "__main__":
    main()