import serial
import time
import pyvisa
import os
import numpy as np


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
STATE_DURATION = 0.2  # Duration to hold each state (seconds)
INTER_STATE_DELAY = 0.1  # Brief delay between state changes (seconds)

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

def send_heater_values(ser, heater_values):
    voltage_message = "".join(f"{heater},{value};" for heater, value in heater_values.items()) + '\n'
    ser.write(voltage_message.encode())
    ser.flush()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.01)

def main():
    try:
        # Initialize heater configuration
        initial_config = {0: 0.1, 1: 0.424, 2: 1.729, 3: 0.1, 4: 2.272, 5: 0.634, 6: 2.414, 7: 0.447, 8: 0.1, 9: 3.76, 10: 0.1, 11: 0.1, 12: 3.033, 13: 0.718, 14: 0.448, 15: 0.431, 16: 0.418, 17: 1.545, 18: 0.198, 19: 2.9, 20: 0.446, 21: 0.723, 22: 0.168, 23: 0.1, 24: 4.841, 25: 0.919, 26: 4.9, 27: 0.1, 28: 1.291, 29: 1.366, 30: 0.862, 31: 1.506, 32: 0.1, 33: 0.01, 34: 0.01, 35: 0.01, 36: 0.01, 37: 0.0, 38: 0.0, 39: 0.01}
        
        heater_values = {int(k): float(v) for k, v in initial_config.items()}
        
        # Initialize hardware
        print("Initializing hardware...")
        scope, ser = init_hardware()
        print("Hardware initialized successfully!")
        
        print(f"Starting continuous sequence: {' -> '.join(SEQUENCE_STATES)} (repeating)")
        print(f"State duration: {STATE_DURATION}s, Inter-state delay: {INTER_STATE_DELAY}s")
        print("Press Ctrl+C to stop\n")
        
        # Start the never-ending sequence
        state_index = 0
        cycle_count = 0
        
        while True:
            # Get current state
            current_state = SEQUENCE_STATES[state_index]
            heater_values[38], heater_values[37] = HEATER_COMBINATIONS[current_state]
            
            # Send heater values
            send_heater_values(ser, heater_values)
            
            # Display current state
            if state_index == 0:
                cycle_count += 1
                print(f"Cycle {cycle_count}:")
            
            print(f"  State {current_state}: Input A={heater_values[38]:.1f}V, Input B={heater_values[37]:.1f}V")
            
            # Hold the state for the specified duration
            time.sleep(STATE_DURATION)
            
            # Brief delay between states
            if INTER_STATE_DELAY > 0:
                time.sleep(INTER_STATE_DELAY)
            
            # Move to next state
            state_index = (state_index + 1) % len(SEQUENCE_STATES)
                
    except KeyboardInterrupt:
        print("\n\nSequence stopped by user (Ctrl+C)")
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