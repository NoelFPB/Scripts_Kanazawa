import serial
import keyboard
import time

# Serial port configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

# Initial configuration for all heaters
INITIAL_CONFIG = {
    0: 0.10, 1: 0.10, 2: 0.10, 3: 0.10, 4: 4.90, 5: 0.10, 6: 0.10,
    7: 4.90, 8: 4.90, 9: 4.90, 10: 4.90, 11: 3.70, 12: 0.10, 13: 0.10,
    14: 4.90, 15: 4.90, 16: 3.70, 17: 0.10, 18: 3.70, 19: 1.00, 20: 0.10,
    21: 0.10, 22: 1.00, 23: 4.90, 24: 3.70, 25: 0.10, 26: 4.90, 27: 4.90,
    28: 4.90, 29: 4.90, 30: 3.70, 31: 2.50, 32: 3.70, 33: 0.01, 34: 0.10,
    35: 0.50, 36: 0.10, 37: 0.01, 38: 0.01, 39: 0.01
}

def init_serial():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    return ser

def send_all_heater_values(ser, heater_values):
    try:
        # Create message for all heaters
        message = "".join(f"{heater},{value};" for heater, value in heater_values.items()) + "\n"
        ser.write(message.encode())
        ser.flush()
        print(f"Set H36: {heater_values[36]}V")
    except Exception as e:
        print(f"Error: {e}")

def main():
    try:
        ser = init_serial()
        print("\nHeater 36 Toggle Control (Sending All Values)")
        print("-----------------------------------------")
        print("SPACE: Toggle H36 between 0.1V and 4.9V")
        print("q: Quit")
        
        # Initialize heater values dictionary
        heater_values = INITIAL_CONFIG.copy()
        h36_state = False  # False = 0.1V, True = 4.9V
         
        while True:
            if keyboard.is_pressed('q'):
                print("\nExiting...")
                break
                
            if keyboard.is_pressed('space'):
                h36_state = not h36_state
                heater_values[36] = 4.9 if h36_state else 0.1
                send_all_heater_values(ser, heater_values)
                time.sleep(0.2)  # Simple debounce
            
            time.sleep(0.01)  # Prevent CPU hogging
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        ser.close()
        print("Serial port closed")

if __name__ == "__main__":
    main()