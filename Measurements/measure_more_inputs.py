import serial
import time
import pyvisa
import keyboard
import os
import numpy as np
import pandas as pd # Import pandas for Excel export

# --- Configuration from Optimization Script ---
# Voltage definitions
V_MIN = 0.1     # Representing logical LOW
V_MAX = 4.9     # Representing logical HIGH

# === MULTI-GATE CONFIGURATION ===
# Define each logic gate you want to test.
# This structure should match the one used in your optimization script.
GATE_CONFIGURATIONS = [
    {
        "name": "OR_GATE_1",
        "type": "OR",
        "input_heaters": [34, 35],  # Heaters for input A and B
        "output_channel": 2         # Oscilloscope channel for this gate's output
    },
    {
        "name": "AND_GATE_1",
        "type": "AND",
        "input_heaters": [36, 37],
        "output_channel": 3
    },
    {
        "name": "NOR_GATE_1",
        "type": "NOR",
        "input_heaters": [38, 39],
        "output_channel": 4
    }
    # Add more gates here as needed, ensuring unique input_heaters (if they are truly independent)
    # and unique output_channel for each.
]

# Heater configuration for the entire chip
FIXED_FIRST_LAYER_RANGE = list(range(33, 40)) # Heaters from 33 to 39 (inclusive)

def generate_truth_table(gate_type, num_inputs, v_min, v_max):
    """Generate truth table for given gate type and number of inputs."""
    input_combinations = []
    for i in range(2**num_inputs):
        binary_representation = bin(i)[2:].zfill(num_inputs)
        combination = tuple(v_max if bit == '1' else v_min for bit in binary_representation)
        input_combinations.append(combination)

    truth_tables_base = {
        "AND": [False] * (2**num_inputs - 1) + [True],
        "OR": [False] + [True] * (2**num_inputs - 1),
        "NAND": [True] * (2**num_inputs - 1) + [False],
        "NOR": [True] + [False] * (2**num_inputs - 1),
    }

    if gate_type in truth_tables_base:
        truth_values = truth_tables_base[gate_type]
    elif gate_type == "XOR":
        xor_results = []
        for combo in input_combinations:
            num_high_inputs = sum(1 for val in combo if val == v_max)
            xor_results.append(num_high_inputs % 2 == 1)
        truth_values = xor_results
    elif gate_type == "XNOR":
        xnor_results = []
        for combo in input_combinations:
            num_high_inputs = sum(1 for val in combo if val == v_max)
            xnor_results.append(num_high_inputs % 2 == 0)
        truth_values = xnor_results
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")
    
    if len(input_combinations) != len(truth_values):
        raise ValueError(f"Truth table length mismatch for {gate_type} with {num_inputs} inputs. Expected {len(input_combinations)}, got {len(truth_values)}")

    return {input_pair: output for input_pair, output in
            zip(input_combinations, truth_values)}
# --- End Configuration from Optimization Script ---


# Serial port configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# Measurement parameters
NUM_MEASUREMENTS = 3    # Number of measurements to average
MEASUREMENT_DELAY = 0.1 # Delay between measurements in seconds

# Performance criteria for automated report
MIN_LOGIC_SEPARATION_VOLTAGE = 0.5 # Minimum voltage difference between min_HIGH and max_LOW
MIN_EXTINCTION_RATIO_DB = 3.0     # Minimum ER in dB for a "passing" gate

def init_hardware(gate_configs):
    """Initializes oscilloscope and serial connection based on gate configurations."""
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    if not resources:
        raise Exception("No VISA resources found. Make sure the oscilloscope is connected.")
    scope = rm.open_resource(resources[0])
    scope.timeout = 5000

    # Initial oscilloscope setup for all necessary channels
    configured_channels = set()
    for gate_info in gate_configs:
        channel = gate_info["output_channel"]
        if channel not in configured_channels: # Avoid reconfiguring same channel
            print(f"Setting up oscilloscope Channel {channel}...")
            scope.write(f':CHANnel{channel}:DISPlay ON')
            scope.write(f':CHANnel{channel}:SCALe 2')
            scope.write(f':CHANnel{channel}:OFFSet -6')
            configured_channels.add(channel)

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2) # Wait for connection to stabilize
    
    return scope, ser

def measure_outputs(scope, channels_to_read):
    """
    Measures outputs from specified oscilloscope channels.
    Args:
        scope: The pyvisa oscilloscope object.
        channels_to_read: A list of channel numbers (e.g., [2, 3, 4]) to measure.
    Returns:
        A dictionary mapping channel number to its average measured voltage.
    """
    try:
        measurements_raw = {ch: [] for ch in channels_to_read}
        
        for i in range(NUM_MEASUREMENTS):
            for ch_num in channels_to_read:
                try:
                    value = float(scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{ch_num}'))
                    measurements_raw[ch_num].append(value)
                except Exception as e:
                    # print(f"Error measuring Channel {ch_num} on pass {i+1}: {e}") # Suppress repetitive error for cleaner output
                    pass # Just don't append, it will result in None for average
            time.sleep(MEASUREMENT_DELAY)
                
        averages = {}
        for ch_num, channel_data in measurements_raw.items():
            if channel_data:
                average = round(sum(channel_data) / len(channel_data), 5)
                averages[ch_num] = average
            else:
                averages[ch_num] = None # Indicate no data for this channel
        
        return averages
    except Exception as e:
        print(f"Error measuring outputs: {e}")
        return {ch: None for ch in channels_to_read}

def send_heater_values(ser, heater_values_dict):
    """
    Sends a complete dictionary of heater values to the hardware.
    Ensures all 40 heaters are always included in the message.
    """
    voltage_message_parts = []
    # Ensure all 40 heaters are in the message, default to 0.0 if not specified
    for i in range(40):
        value = heater_values_dict.get(i, 0.0)
        voltage_message_parts.append(f"{i},{value}")
    voltage_message = ";".join(voltage_message_parts) + '\n'

    ser.write(voltage_message.encode())
    ser.flush()
    time.sleep(0.01) # Small delay for serial buffer
    ser.reset_input_buffer()
    ser.reset_output_buffer()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_main_help():
    print("\n--- Multi-Gate Test Script ---")
    print("Select an option:")
    print(" s: Select a specific logic gate to test (individual gate control)")
    print(" a: Test ALL gates simultaneously (combined input control)")
    print(" r: Run full automated report (test all combined states, then summarize)")
    print(" i: Run individual gate test report (test each gate independently, then summarize)") # New option
    print(" f: Display the full current heater configuration")
    print(" q: Quit")

def print_gate_selection_menu(gates_info):
    print("\n--- Select a Gate to Test Individually ---")
    for idx, gate in enumerate(gates_info):
        print(f" {idx + 1}: {gate['name']} (Inputs: {gate['input_heaters']}, Output: Ch{gate['output_channel']})")
    print(" b: Go back to main menu")
    print(" q: Quit")

def print_gate_test_help(gate_name, num_inputs):
    print(f"\n--- Testing {gate_name} (has {num_inputs} inputs) ---")
    print(" n: Next input combination")
    print(" p: Previous input combination")
    print(" t: Test current input combination (re-measure)")
    print(" s: Show truth table for this gate")
    print(" b: Go back to gate selection menu")
    print(" q: Quit")

def print_all_gates_test_help(num_combinations):
    print(f"\n--- Testing ALL Gates Simultaneously ---")
    print(f"Total combined input states to cycle through: {num_combinations}")
    print(" n: Next combined input state")
    print(" p: Previous combined input state")
    print(" t: Test current combined input state (re-measure)")
    print(" b: Go back to main menu")
    print(" q: Quit")

def display_full_heater_config(heater_values):
    print("\n--- Full Heater Values (H0-H39) ---")
    for i in range(0, 40, 8): # Display in rows of 8
        values = [f"H{j}: {heater_values.get(j, 0.0):.3f}V" for j in range(i, min(i+8, 40))]
        print("    ".join(values))
    print("\n")

def get_all_input_heaters(gates_info):
    """Collects all unique input heaters from all defined gates."""
    all_input_heaters = set()
    for gate in gates_info:
        all_input_heaters.update(gate['input_heaters'])
    return sorted(list(all_input_heaters))

def generate_all_combined_input_states(gates_info):
    """
    Generates all possible combined input states for all unique input heaters.
    Each state is a dictionary: {heater_id: voltage_value, ...}
    """
    all_input_heaters = get_all_input_heaters(gates_info)
    num_unique_inputs = len(all_input_heaters)
    
    if num_unique_inputs == 0:
        return []

    combined_states = []
    # Generate all 2^N combinations for N unique input heaters
    for i in range(2**num_unique_inputs):
        binary_representation = bin(i)[2:].zfill(num_unique_inputs)
        
        current_combined_state = {}
        for j, heater_id in enumerate(all_input_heaters):
            current_combined_state[heater_id] = V_MAX if binary_representation[j] == '1' else V_MIN
        combined_states.append(current_combined_state)
    
    return combined_states

def run_automated_report(scope, ser, current_heater_state_base, gates_info, all_combined_input_states, channels_to_monitor):
    """
    Runs a full automated test, collects data, and generates a report.
    This version tests ALL gates simultaneously across all combinations.
    """
    print("\n--- Running Automated Full Test and Report (Combined Inputs) ---")
    print(f"Testing {len(all_combined_input_states)} combined input states...")
    
    gate_results = {} # Store collected outputs for each gate
    for gate in gates_info:
        gate_results[gate['name']] = {
            'high_outputs': [],
            'low_outputs': [],
            'correct_outputs_count': 0,
            'total_test_combinations': 0,
        }
    
    # This list will hold the rows for the overall detailed table to be exported to Excel
    excel_data_rows = []
    
    # Prepare header for the Excel table
    excel_header_row = ["Test Case"]
    for h_id in get_all_input_heaters(gates_info):
        excel_header_row.append(f"Input H{h_id} (V)")
    for gate in gates_info:
        excel_header_row.append(f"{gate['name']} (Ch{gate['output_channel']}) Measured (V)")
        excel_header_row.append(f"{gate['name']} (Ch{gate['output_channel']}) Expected")
        excel_header_row.append(f"{gate['name']} Status (Pass/Fail)") 

    excel_data_rows.append(excel_header_row)

    # Iterate through all combined input states
    for i, combined_input_heater_values in enumerate(all_combined_input_states):
        print(f"  Testing combined state {i+1}/{len(all_combined_input_states)}...", end='\r')
        # Prepare the full heater state for this test
        test_heater_state = current_heater_state_base.copy()
        test_heater_state.update(combined_input_heater_values)

        send_heater_values(ser, test_heater_state)
        time.sleep(0.2) # Allow values to settle

        current_outputs_measured = measure_outputs(scope, channels_to_monitor)
        
        # Prepare row for the Excel table
        current_excel_row = [f"Combined State {i+1}"]
        
        # Add individual input heater values to the Excel row
        for h_id in get_all_input_heaters(gates_info):
            current_excel_row.append(combined_input_heater_values[h_id])


        # Process results for each gate
        for gate in gates_info:
            gate_name = gate['name']
            output_channel = gate['output_channel']
            
            measured_output = current_outputs_measured.get(output_channel)
            
            # Determine input values for this specific gate based on the combined state
            gate_input_values_tuple = tuple(test_heater_state[h] for h in gate['input_heaters'])
            expected_output_bool = gate['truth_table'].get(gate_input_values_tuple)

            expected_logic_str = "N/A"
            is_correct_combo = False # Per-combination logical correctness
            measured_output_val_for_excel = measured_output # Store raw float for excel


            if measured_output is not None:
                if expected_output_bool is not None:
                    expected_logic_str = 'HIGH' if expected_output_bool else 'LOW'
                    # Determine if measured output matches expected logic (simple midpoint thresholding)
                    is_measured_high = measured_output > (V_MIN + V_MAX) / 2 # Simple midpoint threshold
                    is_correct_combo = (expected_output_bool and is_measured_high) or (not expected_output_bool and not is_measured_high)
            
            # Update gate_results for overall metrics
            gate_results[gate_name]['total_test_combinations'] += 1
            if measured_output is not None:
                if expected_output_bool: # Expected HIGH
                    gate_results[gate_name]['high_outputs'].append(measured_output)
                else: # Expected LOW
                    gate_results[gate_name]['low_outputs'].append(measured_output)
                if is_correct_combo:
                    gate_results[gate_name]['correct_outputs_count'] += 1

            # Add to detailed table row for Excel
            current_excel_row.append(measured_output_val_for_excel)
            current_excel_row.append(expected_logic_str)
            current_excel_row.append("PASS" if is_correct_combo else "FAIL")
        
        excel_data_rows.append(current_excel_row) # Add completed row to Excel data


    # --- Export to Excel ---
    report_filename = "multi_gate_combined_test_report.xlsx"
    try:
        df = pd.DataFrame(excel_data_rows[1:], columns=excel_data_rows[0])
        df.to_excel(report_filename, index=False)
        print(f"\n\n--- Detailed combined test report saved to {report_filename} ---")
    except Exception as e:
        print(f"\n\nERROR: Could not save Excel report: {e}")
        print("Please ensure 'openpyxl' is installed (`pip install openpyxl`) and the file is not open.")


    # --- Print Summary Section (as before, but less verbose in console) ---
    print("\n\n--- Automated Report Summary (Performance Metrics) ---")
    overall_gates_working = 0
    
    for gate_name, results in gate_results.items():
        print(f"\n--- Gate: {gate_name} ---")
        
        if not results['high_outputs'] or not results['low_outputs']:
            print("  Insufficient data (missing valid HIGH/LOW measurements) for full metric assessment.")
            print(f"  Logical Correctness Rate: {results['correct_outputs_count']}/{results['total_test_combinations']} ({ (results['correct_outputs_count']/results['total_test_combinations']*100):.1f}%)")
            continue # Skip ER/Separation if data is incomplete

        min_high = min(results['high_outputs'])
        max_low = max(results['low_outputs'])
        logic_separation = min_high - max_low

        er_db = -float('inf')
        if max_low > 0.001 and min_high > 0.001: # Avoid division by zero/near-zero
            er_linear = min_high / max_low
            er_db = 10 * np.log10(er_linear)
        
        print(f"  Min Measured HIGH: {min_high:.3f}V, Max Measured LOW: {max_low:.3f}V")
        print(f"  Logic Separation (Min HIGH - Max LOW): {logic_separation:.3f}V")
        if er_db != -float('inf'):
            print(f"  Extinction Ratio: {er_db:.2f} dB")
        else:
            print("  Extinction Ratio: N/A (Max LOW too close to zero or Min HIGH too close to zero)")

        # --- Pass/Fail Assessment ---
        is_logic_separated = logic_separation > MIN_LOGIC_SEPARATION_VOLTAGE
        is_er_sufficient = er_db >= MIN_EXTINCTION_RATIO_DB
        is_all_combinations_correct = (results['total_test_combinations'] > 0 and 
                                       (results['correct_outputs_count'] / results['total_test_combinations']) == 1.0)
        

        print(f"\n  Overall Gate Status:")
        if is_logic_separated and is_er_sufficient and is_all_combinations_correct:
            print(f"  Status: ✅ PASSED (Meets all performance criteria)")
            overall_gates_working += 1
        else:
            print(f"  Status: ❌ FAILED")
            if not is_logic_separated: 
                print(f"    - Logic separation ({logic_separation:.3f}V) below threshold ({MIN_LOGIC_SEPARATION_VOLTAGE}V)")
            if not is_er_sufficient: 
                print(f"    - Extinction Ratio ({er_db:.2f}dB) below threshold ({MIN_EXTINCTION_RATIO_DB}dB)")
            if not is_all_combinations_correct: 
                print(f"    - Not all ({results['correct_outputs_count']}/{results['total_test_combinations']}) combinations produced correct logical output based on midpoint threshold.")

        print(f"  Logical Correctness Rate: {results['correct_outputs_count']}/{results['total_test_combinations']} ({ (results['correct_outputs_count']/results['total_test_combinations']*100):.1f}%)")

    print(f"\n--- Overall Test Result ---")
    print(f"Total Gates Configured: {len(gates_info)}")
    print(f"Gates Passing All Criteria: {overall_gates_working}")
    if overall_gates_working == len(gates_info):
        print("Conclusion: 🎉 All configured logic gates are performing correctly!")
    else:
        print("Conclusion: ⚠️ Some gates did not pass all performance criteria.")

    input("\nPress Enter to return to main menu...") # Pause for user to read report

# --- New Function for Individual Gate Test Report ---
def run_individual_gate_test_report(scope, ser, current_heater_state_base, gates_info, channels_to_monitor):
    """
    Tests each gate individually by cycling its inputs, keeping other input heaters at V_MIN.
    Generates a separate Excel sheet for each gate.
    """
    print("\n--- Running Automated Individual Gate Test Report ---")
    
    report_filename = "individual_gate_test_report.xlsx"
    
    # Use ExcelWriter to write to multiple sheets in one file
    try:
        with pd.ExcelWriter(report_filename, engine='openpyxl') as writer:
            overall_gates_working_count = 0
            
            for gate_index, gate in enumerate(gates_info):
                gate_name = gate['name']
                input_heaters = gate['input_heaters']
                output_channel = gate['output_channel']
                truth_table = gate['truth_table']
                input_combinations = list(truth_table.keys())

                print(f"\nTesting {gate_name} (Ch{output_channel}) individually...")

                excel_data_rows = []
                # Header for this gate's sheet
                header_row = [f"{gate_name} Input State"]
                for h_id in input_heaters:
                    header_row.append(f"Input H{h_id} (V)")
                header_row.append("Measured Output (V)")
                header_row.append("Expected Logic")
                header_row.append("Status (Pass/Fail)")
                excel_data_rows.append(header_row)

                gate_high_outputs = []
                gate_low_outputs = []
                gate_correct_combinations = 0
                gate_total_combinations = len(input_combinations)

                # Temporarily set all *other* gate inputs to V_MIN
                # Start with the base config and then override only the current gate's inputs
                test_heater_state_for_individual_gate = current_heater_state_base.copy()
                all_input_heaters_across_chip = get_all_input_heaters(gates_info)
                for h_id in all_input_heaters_across_chip:
                    if h_id not in input_heaters: # If it's an input heater for another gate
                        test_heater_state_for_individual_gate[h_id] = V_MIN # Set to LOW

                for i, input_combo in enumerate(input_combinations):
                    print(f"  Testing {gate_name} combo {i+1}/{len(input_combinations)}...", end='\r')
                    
                    current_heater_state_for_combo = test_heater_state_for_individual_gate.copy()
                    for j, h_id in enumerate(input_heaters):
                        current_heater_state_for_combo[h_id] = input_combo[j]

                    send_heater_values(ser, current_heater_state_for_combo)
                    time.sleep(0.2)

                    measured_outputs_all_channels = measure_outputs(scope, channels_to_monitor)
                    measured_output = measured_outputs_all_channels.get(output_channel)

                    expected_bool = truth_table.get(input_combo)
                    
                    # Prepare row for Excel
                    current_excel_row = [f"Combo {i+1}"]
                    for h_id in input_heaters:
                        current_excel_row.append(input_combo[input_heaters.index(h_id)]) # Use index to get voltage from tuple
                    
                    expected_logic_str = "N/A"
                    is_correct_combo = False
                    measured_output_val_for_excel = measured_output

                    if measured_output is not None:
                        if expected_bool is not None:
                            expected_logic_str = 'HIGH' if expected_bool else 'LOW'
                            is_measured_high = measured_output > (V_MIN + V_MAX) / 2
                            is_correct_combo = (expected_bool and is_measured_high) or (not expected_bool and not is_measured_high)
                            
                        if expected_bool:
                            gate_high_outputs.append(measured_output)
                        else:
                            gate_low_outputs.append(measured_output)
                        
                        if is_correct_combo:
                            gate_correct_combinations += 1
                    
                    current_excel_row.append(measured_output_val_for_excel)
                    current_excel_row.append(expected_logic_str)
                    current_excel_row.append("PASS" if is_correct_combo else "FAIL")
                    excel_data_rows.append(current_excel_row)
                
                # Write this gate's data to a new sheet
                df_gate = pd.DataFrame(excel_data_rows[1:], columns=excel_data_rows[0])
                df_gate.to_excel(writer, sheet_name=gate_name, index=False)

                # --- Summarize performance for this individual gate ---
                print(f"\n  --- Summary for {gate_name} ---")
                if not gate_high_outputs or not gate_low_outputs:
                    print("    Insufficient data for full metric assessment (missing HIGH/LOWs or measurement failures).")
                    print(f"    Logical Correctness Rate: {gate_correct_combinations}/{gate_total_combinations} ({ (gate_correct_combinations/gate_total_combinations*100):.1f}%)")
                    continue

                min_high = min(gate_high_outputs)
                max_low = max(gate_low_outputs)
                logic_separation = min_high - max_low

                er_db = -float('inf')
                if max_low > 0.001 and min_high > 0.001:
                    er_linear = min_high / max_low
                    er_db = 10 * np.log10(er_linear)
                
                print(f"    Min Measured HIGH: {min_high:.3f}V, Max Measured LOW: {max_low:.3f}V")
                print(f"    Logic Separation: {logic_separation:.3f}V")
                if er_db != -float('inf'):
                    print(f"    Extinction Ratio: {er_db:.2f} dB")

                is_logic_separated = logic_separation > MIN_LOGIC_SEPARATION_VOLTAGE
                is_er_sufficient = er_db >= MIN_EXTINCTION_RATIO_DB
                is_all_combinations_correct = (gate_total_combinations > 0 and 
                                               (gate_correct_combinations / gate_total_combinations) == 1.0)
                
                if is_logic_separated and is_er_sufficient and is_all_combinations_correct:
                    print(f"    Status: ✅ PASSED (Meets all performance criteria individually)")
                    overall_gates_working_count += 1
                else:
                    print(f"    Status: ❌ FAILED")
                    if not is_logic_separated: print(f"      - Logic separation ({logic_separation:.3f}V) below threshold ({MIN_LOGIC_SEPARATION_VOLTAGE}V)")
                    if not is_er_sufficient: print(f"      - Extinction Ratio ({er_db:.2f}dB) below threshold ({MIN_EXTINCTION_RATIO_DB}dB)")
                    if not is_all_combinations_correct: print(f"      - Not all ({gate_correct_combinations}/{gate_total_combinations}) combinations produced correct logical output.")
                print(f"    Logical Correctness Rate: {gate_correct_combinations}/{gate_total_combinations} ({ (gate_correct_combinations/gate_total_combinations*100):.1f}%)")

            # Restore all input heaters to their base_config values after individual tests
            for h_id in all_input_heaters_across_chip:
                current_heater_state_base[h_id] = current_heater_state_base.get(h_id, 0.0)
            send_heater_values(ser, current_heater_state_base)
            time.sleep(0.1)

        print(f"\n\n--- Individual Gate Test Report Saved to {report_filename} ---")
        print(f"\n--- Overall Individual Test Summary ---")
        print(f"Total Gates Configured: {len(gates_info)}")
        print(f"Gates Passing All Individual Criteria: {overall_gates_working_count}")
        if overall_gates_working_count == len(gates_info):
            print("Conclusion: 🎉 All configured logic gates are performing correctly when tested individually!")
        else:
            print("Conclusion: ⚠️ Some gates did not pass all individual performance criteria.")
        
    except Exception as e:
        print(f"\n\nERROR: Could not save individual Excel report: {e}")
        print("Please ensure 'openpyxl' is installed (`pip install openpyxl`) and the file is not open.")

    input("\nPress Enter to return to main menu...")


def main():
    scope, ser = None, None # Initialize to None for cleanup in finally block
    
    try:
        # Define the optimized configuration you want to test
        # This should be the 'complete_config' dictionary from your optimization script's output
        # Example: Replace this with the actual output from your optimization run
        optimized_heater_config =  {0: 4.9, 1: 0.601, 2: 0.85, 3: 3.034, 4: 1.679, 5: 0.838, 6: 2.429, 7: 0.1, 8: 0.5, 9: 4.696, 10: 0.302, 11: 4.079, 12: 1.218, 13: 4.763, 14: 1.329, 15: 0.1, 16: 2.922, 17: 3.183, 18: 3.077, 19: 0.657, 20: 4.105, 21: 3.178, 22: 2.605, 23: 1.894, 24: 4.765, 25: 0.938, 26: 0.1, 27: 1.037, 28: 4.816, 29: 0.248, 30: 2.536, 31: 2.809, 32: 4.263, 33: 0.01, 34: 0.0, 35: 0.0, 36: 0.0, 37: 0.0, 38: 0.0, 39: 0.0}

        # Ensure all 40 heaters are present, default to 0.0 if not in your optimized_heater_config
        # This will be our base "background" configuration for the non-input heaters.
        # This dictionary will be modified for inputs during tests, but reset to these base values.
        current_heater_state_base = {i: optimized_heater_config.get(i, 0.0) for i in range(40)}

        scope, ser = init_hardware(GATE_CONFIGURATIONS)

        # Pre-process gate data for easy access and consolidate channels to monitor
        gates_info = []
        channels_to_monitor = set()
        for gate_config in GATE_CONFIGURATIONS:
            gate_data = {
                "name": gate_config["name"],
                "type": gate_config["type"],
                "input_heaters": gate_config["input_heaters"],
                "output_channel": gate_config["output_channel"],
                "truth_table": generate_truth_table(gate_config["type"], len(gate_config["input_heaters"]), V_MIN, V_MAX)
            }
            gates_info.append(gate_data)
            channels_to_monitor.add(gate_config["output_channel"])
        
        # Convert to a sorted list for consistent measurement order
        channels_to_monitor = sorted(list(channels_to_monitor))

        # Generate all combined input states for the "Test ALL gates" mode
        all_combined_input_states = generate_all_combined_input_states(gates_info)
        
        mode = "main_menu" # State variable: "main_menu", "individual_gate_selection", "individual_gate_test", "all_gates_test", "automated_report", "individual_report"
        active_gate_idx = -1 # For individual gate test
        input_combo_idx = 0  # For individual and all-gates test

        # Send the initial optimized base configuration to the chip
        send_heater_values(ser, current_heater_state_base)
        time.sleep(0.1)

        while True:
            clear_screen()
            if mode == "main_menu":
                print_main_help()
                key_pressed = keyboard.read_event(suppress=True).name
                if key_pressed == 's':
                    mode = "individual_gate_selection"
                elif key_pressed == 'a':
                    mode = "all_gates_test"
                    input_combo_idx = 0 # Reset for this mode
                elif key_pressed == 'r':
                    mode = "automated_report"
                elif key_pressed == 'i': # New option for individual report
                    mode = "individual_report"
                elif key_pressed == 'f':
                    display_full_heater_config(current_heater_state_base)
                    time.sleep(1) # Pause to allow user to read
                elif key_pressed == 'q':
                    break
            
            elif mode == "individual_gate_selection":
                print_gate_selection_menu(gates_info)
                key_pressed = keyboard.read_event(suppress=True).name
                if key_pressed == 'b':
                    mode = "main_menu"
                elif key_pressed == 'q':
                    break
                elif key_pressed.isdigit() and 1 <= int(key_pressed) <= len(gates_info):
                    active_gate_idx = int(key_pressed) - 1
                    input_combo_idx = 0 # Reset combo index for new gate
                    mode = "individual_gate_test"
                    # Restore base_config for input heaters as they will be manipulated
                    # This ensures individual test starts from a clean slate for inputs
                    for h_id in gates_info[active_gate_idx]['input_heaters']:
                        current_heater_state_base[h_id] = optimized_heater_config.get(h_id, 0.0)
                    send_heater_values(ser, current_heater_state_base) # Send base config for gate inputs
                    time.sleep(0.1)
                else:
                    print("Invalid selection. Please try again.")
                    time.sleep(0.5)

            elif mode == "individual_gate_test":
                current_gate = gates_info[active_gate_idx]
                truth_table_keys = list(current_gate["truth_table"].keys())
                
                if not truth_table_keys:
                    print(f"Error: No input combinations found for {current_gate['name']}. Returning to gate selection.")
                    mode = "individual_gate_selection"
                    time.sleep(2)
                    continue

                print_gate_test_help(current_gate["name"], len(current_gate["input_heaters"]))
                
                current_input_combination = truth_table_keys[input_combo_idx]
                expected_output_logic = current_gate["truth_table"][current_input_combination]

                print(f"\nGate: {current_gate['name']} (Ch{current_gate['output_channel']})")
                print(f"Current Input Combination ({input_combo_idx + 1}/{len(truth_table_keys)}):")
                # Apply the current input combination to the heaters
                test_heater_state = current_heater_state_base.copy() # Start with the base config
                for i, heater in enumerate(current_gate["input_heaters"]):
                    test_heater_state[heater] = current_input_combination[i]
                    print(f"  Heater {heater}: {current_input_combination[i]:.3f}V")
                print(f"Expected Output: {'HIGH' if expected_output_logic else 'LOW'}")

                send_heater_values(ser, test_heater_state)
                time.sleep(0.2) # Allow values to settle

                current_outputs = measure_outputs(scope, channels_to_monitor)
                print("\nMeasured Outputs:")
                for ch_num in channels_to_monitor:
                    output_val = current_outputs.get(ch_num)
                    gate_for_channel = next((g for g in gates_info if g["output_channel"] == ch_num), None)
                    gate_label = f" ({gate_for_channel['name']})" if gate_for_channel else ""
                    if output_val is not None:
                        print(f"  Channel {ch_num}{gate_label}: {output_val:.4f}V")
                    else:
                        print(f"  Channel {ch_num}{gate_label}: N/A (Measurement Failed)")


                key_pressed = keyboard.read_event(suppress=True).name
                if key_pressed == 'n':
                    input_combo_idx = (input_combo_idx + 1) % len(truth_table_keys)
                elif key_pressed == 'p':
                    input_combo_idx = (input_combo_idx - 1 + len(truth_table_keys)) % len(truth_table_keys)
                elif key_pressed == 't':
                    pass # Loop will re-evaluate
                elif key_pressed == 's':
                    print(f"\n--- Truth Table for {current_gate['name']} ---")
                    for combo, expected_val in current_gate["truth_table"].items():
                        input_heater_values = ", ".join(f"H{h}={combo[i]:.1f}V" for i,h in enumerate(current_gate['input_heaters']))
                        print(f"  Inputs ({input_heater_values}) -> Expected: {'HIGH' if expected_val else 'LOW'}")
                    time.sleep(3) # Pause to allow user to read
                elif key_pressed == 'b':
                    mode = "main_menu" # Back to main menu (or "individual_gate_selection")
                    # Restore base_config for input heaters, they shouldn't stick at last test value
                    for h_id in current_gate['input_heaters']:
                        current_heater_state_base[h_id] = optimized_heater_config.get(h_id, 0.0)
                    send_heater_values(ser, current_heater_state_base)
                    time.sleep(0.1)
                elif key_pressed == 'q':
                    break
                else:
                    print("Invalid input. Use n, p, t, s, b, or q.")
                    time.sleep(0.5)

            elif mode == "all_gates_test":
                print_all_gates_test_help(len(all_combined_input_states))
                
                # Get the current combined input state from the generated list
                current_combined_input_heater_values = all_combined_input_states[input_combo_idx]

                # Prepare the full heater state for this test
                test_heater_state = current_heater_state_base.copy()
                test_heater_state.update(current_combined_input_heater_values)

                print(f"\n--- Combined Input State ({input_combo_idx + 1}/{len(all_combined_input_states)}) ---")
                # Display inputs for each gate for clarity
                for gate in gates_info:
                    gate_inputs = []
                    for i, heater_id in enumerate(gate['input_heaters']):
                        gate_inputs.append(f"H{heater_id}={test_heater_state[heater_id]:.3f}V")
                    print(f"  {gate['name']} Inputs: ({', '.join(gate_inputs)})")

                send_heater_values(ser, test_heater_state)
                time.sleep(0.2) # Allow values to settle

                current_outputs = measure_outputs(scope, channels_to_monitor)
                print("\nMeasured Outputs:")
                for ch_num in channels_to_monitor:
                    output_val = current_outputs.get(ch_num)
                    gate_for_channel = next((g for g in gates_info if g["output_channel"] == ch_num), None)
                    gate_label = f" ({gate_for_channel['name']})" if gate_for_channel else ""
                    
                    # Also determine expected logical output for the gate
                    expected_logic = "N/A"
                    if gate_for_channel:
                        # Construct the input combination tuple for this specific gate from the current_combined_input_heater_values
                        gate_input_tuple = tuple(test_heater_state[h] for h in gate_for_channel['input_heaters'])
                        expected_bool = gate_for_channel['truth_table'].get(gate_input_tuple)
                        if expected_bool is not None:
                            expected_logic = 'HIGH' if expected_bool else 'LOW'

                    if output_val is not None:
                        print(f"  Channel {ch_num}{gate_label}: {output_val:.4f}V (Expected: {expected_logic})")
                    else:
                        print(f"  Channel {ch_num}{gate_label}: N/A (Measurement Failed)")
                
                key_pressed = keyboard.read_event(suppress=True).name
                if key_pressed == 'n':
                    input_combo_idx = (input_combo_idx + 1) % len(all_combined_input_states)
                elif key_pressed == 'p':
                    input_combo_idx = (input_combo_idx - 1 + len(all_combined_input_states)) % len(all_combined_input_states)
                elif key_pressed == 't':
                    pass # Loop will re-evaluate
                elif key_pressed == 'b':
                    mode = "main_menu"
                    # Restore base_config for input heaters
                    for h_id in get_all_input_heaters(gates_info):
                        current_heater_state_base[h_id] = optimized_heater_config.get(h_id, 0.0)
                    send_heater_values(ser, current_heater_state_base)
                    time.sleep(0.1)
                elif key_pressed == 'q':
                    break
                else:
                    print("Invalid input. Use n, p, t, b, or q.")
                    time.sleep(0.5)

            elif mode == "automated_report":
                run_automated_report(scope, ser, current_heater_state_base, gates_info, all_combined_input_states, channels_to_monitor)
                mode = "main_menu" # Return to main menu after report
            
            elif mode == "individual_report": # New mode for individual gate test report
                run_individual_gate_test_report(scope, ser, current_heater_state_base, gates_info, channels_to_monitor)
                mode = "main_menu" # Return to main menu after report

            time.sleep(0.1) # Small delay to prevent busy looping

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
    finally:
        # Ensure hardware resources are closed even if an.error occurs
        if ser and ser.is_open:
            send_heater_values(ser, {i: 0.0 for i in range(40)}) # Reset all heaters to 0V
            time.sleep(0.2)
            ser.close()
            print("Serial connection closed.")
        if scope:
            scope.close()
            print("Oscilloscope connection closed.")
        print("\nProgram terminated.")

if __name__ == "__main__":
    main()