import serial
import time
import pyvisa
import os
import numpy as np
import csv
from datetime import datetime


# USING as input 36 and 37
# Serial port configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# Define heater combinations for each state
HEATER_COMBINATIONS = {
    '00': (0.1, 0.1),  # State 00
    '01': (0.1, 4.9),  # State 01
    '10': (4.9, 0.1),  # State 10
    '11': (4.9, 4.9)   # State 11
}

# Sequence configuration
SEQUENCE_STATES = ['00', '01', '10', '11']  # Order of states to cycle through
STATE_DURATION = 0.5  # Duration to hold each state (seconds)
INTER_STATE_DELAY = 0.5  # Brief delay between state changes (seconds)

# Data logging configuration
SAMPLE_RATE = 20  # Samples per second
SAMPLES_PER_STATE = int(STATE_DURATION * SAMPLE_RATE)  # Number of samples per state
LOG_FILENAME = f"scope_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Channel selection - set to True for channels you want to measure and log
CHANNELS_TO_MEASURE = [True, True, True, True]  # CH1, CH2, CH3, CH4

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

def measure_single_point(scope):
    """Measure a single data point from all active channels"""
    measurements = {}
    channel_names = ['CHANnel1', 'CHANnel2', 'CHANnel3', 'CHANnel4']
    
    try:
        for ch_idx, is_active in enumerate(CHANNELS_TO_MEASURE):
            if is_active:
                # You can change this measurement type as needed:
                # VMAX, VMIN, VPP, VRMS, VAVerage, etc.
                value = float(scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,{channel_names[ch_idx]}'))
                measurements[f'CH{ch_idx+1}'] = value
            else:
                measurements[f'CH{ch_idx+1}'] = None
    except Exception as e:
        print(f"Error measuring: {e}")
        for ch_idx in range(4):
            measurements[f'CH{ch_idx+1}'] = None
    
    return measurements

def send_heater_values(ser, heater_values):
    voltage_message = "".join(f"{heater},{value};" for heater, value in heater_values.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.01)

def setup_csv_file(filename):
    """Initialize CSV file with headers"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'cycle', 'state', 'input_A', 'input_B', 'sample_in_state']
        
        # Add channel headers for active channels
        for ch_idx, is_active in enumerate(CHANNELS_TO_MEASURE):
            if is_active:
                fieldnames.append(f'CH{ch_idx+1}')
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
    return fieldnames

def log_data_point(filename, fieldnames, timestamp, cycle, state, input_a, input_b, sample_num, measurements):
    """Log a single data point to CSV"""
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        row = {
            'timestamp': timestamp,
            'cycle': cycle,
            'state': state,
            'input_A': input_a,
            'input_B': input_b,
            'sample_in_state': sample_num
        }
        
        # Add channel measurements
        for ch_idx, is_active in enumerate(CHANNELS_TO_MEASURE):
            if is_active:
                row[f'CH{ch_idx+1}'] = measurements.get(f'CH{ch_idx+1}')
        
        writer.writerow(row)

def main():
    try:
        # Initialize heater configuration
        initial_config = {0: 0.1, 1: 0.424, 2: 1.729, 3: 0.1, 4: 2.272, 5: 0.634, 6: 2.414, 7: 0.447, 8: 0.1, 9: 3.76, 10: 0.1, 11: 0.1, 12: 3.033, 13: 0.718, 14: 0.448, 15: 0.431, 16: 0.418, 17: 1.545, 18: 0.198, 19: 2.9, 20: 0.446, 21: 0.723, 22: 0.168, 23: 0.1, 24: 4.841, 25: 0.919, 26: 4.9, 27: 0.1, 28: 1.291, 29: 1.366, 30: 0.862, 31: 1.506, 32: 0.1, 33: 0.01, 34: 0.01, 35: 0.01, 36: 0.01, 37: 0.0, 38: 0.0, 39: 0.01}
        
        heater_values = {int(k): float(v) for k, v in initial_config.items()}
        
        # Initialize hardware
        print("Initializing hardware...")
        scope, ser = init_hardware()
        print("Hardware initialized successfully!")
        
        # Setup CSV logging
        fieldnames = setup_csv_file(LOG_FILENAME)
        print(f"Data will be saved to: {LOG_FILENAME}")
        
        print(f"Starting continuous sequence with data logging:")
        print(f"Sequence: {' -> '.join(SEQUENCE_STATES)} (repeating)")
        print(f"State duration: {STATE_DURATION}s, Sample rate: {SAMPLE_RATE} Hz")
        print(f"Samples per state: {SAMPLES_PER_STATE}")
        
        active_channels = [f"CH{i+1}" for i, active in enumerate(CHANNELS_TO_MEASURE) if active]
        print(f"Logging channels: {', '.join(active_channels)}")
        print("Press Ctrl+C to stop\n")
        
        # Start the never-ending sequence with data logging
        state_index = 0
        cycle_count = 0
        start_time = time.time()
        
        while True:
            # Get current state
            current_state = SEQUENCE_STATES[state_index]
            heater_values[38], heater_values[37] = HEATER_COMBINATIONS[current_state]
            
            print(heater_values)
            print("")
            # Send heater values
            send_heater_values(ser, heater_values)
            
            # Display current state
            if state_index == 0:
                cycle_count += 1
                print(f"Cycle {cycle_count}:")
            
            print(f"  State {current_state}: Input A={heater_values[38]:.1f}V, Input B={heater_values[37]:.1f}V", end="")
            
            # Sample data during this state
            state_start_time = time.time()
            sample_interval = 1.0 / SAMPLE_RATE  # Time between samples
            
            for sample_num in range(SAMPLES_PER_STATE):
                sample_time = state_start_time + (sample_num * sample_interval)
                
                # Wait until it's time for the next sample
                while time.time() < sample_time:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                
                # Take measurement
                measurements = measure_single_point(scope)
                
                # Log the data
                timestamp = time.time() - start_time  # Relative timestamp
                log_data_point(
                    LOG_FILENAME, fieldnames, timestamp, cycle_count, 
                    current_state, heater_values[38], heater_values[37], 
                    sample_num + 1, measurements
                )
            
            # Show sample of the measured values
            if any(measurements.values()):
                print(" | Outputs:", end="")
                for ch_name, value in measurements.items():
                    if value is not None:
                        print(f" {ch_name}:{value:.3f}V", end="")
            print()
            
            # Brief delay between states
            if INTER_STATE_DELAY > 0:
                time.sleep(INTER_STATE_DELAY)
            
            # Move to next state
            state_index = (state_index + 1) % len(SEQUENCE_STATES)
                
    except KeyboardInterrupt:
        print(f"\n\nSequence stopped by user (Ctrl+C)")
        print(f"Data saved to: {LOG_FILENAME}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'ser' in locals():
            ser.close()
        if 'scope' in locals():
            scope.close()
        print("Hardware connections closed.")
        print("Program terminated.")

if __name__ == "__main__":
    main()