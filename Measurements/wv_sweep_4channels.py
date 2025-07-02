import serial
import time
import pyvisa
import numpy as np
import pandas as pd
from datetime import datetime
import csv

# Serial port configuration 
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# Hardware addresses
LASER_ADDRESS = "GPIB0::6::INSTR"

# Input heaters and voltage definitions
INPUT_HEATERS = [36, 37]  # Heaters for input A and B
V_MIN = 0.1    # Representing logical LOW (0)
V_MAX = 4.9    # Representing logical HIGH (1)

# Wavelength sweep parameters
WAVELENGTH_START = 1548.0  # nm
WAVELENGTH_END = 1552.0    # nm
WAVELENGTH_STEP = 0.4      # nm

# Input combinations for testing
INPUT_COMBINATIONS = [
    (V_MIN, V_MIN),  # 00
    (V_MIN, V_MAX),  # 01
    (V_MAX, V_MIN),  # 10
    (V_MAX, V_MAX)   # 11
]

INPUT_LABELS = ["00", "01", "10", "11"]

# Gate type mapping for clean output
GATE_TYPE_MAPPING = {
    "00": "Gate_00",
    "01": "Gate_01",
    "10": "Gate_10", 
    "11": "Gate_11"
}

# Channel configuration - customize these labels based on your setup
CHANNEL_CONFIG = {
    1: {"label": "Output_1", "display": True},
    2: {"label": "Output_2", "display": True}, 
    3: {"label": "Output_3", "display": True},
    4: {"label": "Output_4", "display": True}
}

# Measurement parameters
NUM_MEASUREMENTS_PER_POINT = 3  # Number of measurements to average (no std dev displayed)
LASER_SETTLING_TIME = 1       # Time to wait after wavelength change (seconds)
LASER_STARTUP_TIME = 15       # Time to wait for laser to turn on and stabilize (seconds)
HEATER_SETTLING_TIME = 0.3    # Time to wait after heater change (seconds)
FINAL_SETTLING_TIME = 0.1     # Additional settling before measurements (seconds)

class WavelengthSweepMeasurement:
    def __init__(self, base_config=None):   
        """
        Initialize the measurement system
        
        Args:
            base_config: Dictionary of base heater values. If None, uses a minimal config.
        """
        # Use provided base config or create a minimal one
        if base_config is None:
            # Minimal configuration - all heaters at low values except inputs
            self.base_config = {i: 0.01 for i in range(40)}
            # Input heaters will be overridden during measurement
            self.base_config[36] = 0.0  # Will be set per measurement
            self.base_config[37] = 0.0  # Will be set per measurement
        else:
            self.base_config = base_config.copy()
        
        # Initialize hardware
        self.scope = self._init_scope()
        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        self.laser = self._init_laser()
        time.sleep(2)  # Wait for connections to stabilize
        
        # Storage for results
        self.results = []
        
        # Get active channels for display
        active_channels = [ch for ch, config in CHANNEL_CONFIG.items() if config["display"]]
        
        print(f"Wavelength sweep initialized:")
        print(f"  Range: {WAVELENGTH_START}nm to {WAVELENGTH_END}nm")
        print(f"  Step size: {WAVELENGTH_STEP}nm")
        print(f"  Total wavelengths: {len(self.get_wavelength_list())}")
        print(f"  Input combinations: {len(INPUT_COMBINATIONS)}")
        print(f"  Active channels: {active_channels}")
        print(f"  Total measurements: {len(self.get_wavelength_list()) * len(INPUT_COMBINATIONS) * len(active_channels)}")
    
    def get_wavelength_list(self):
        """Generate list of wavelengths to measure"""
        wavelengths = []
        current = WAVELENGTH_START
        while current <= WAVELENGTH_END + 1e-6:  # Small epsilon for floating point comparison
            wavelengths.append(round(current, 1))
            current += WAVELENGTH_STEP
        return wavelengths
    
    def _init_scope(self):
        """Initialize oscilloscope for all 4 channels"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No scope VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
        
        # Setup all 4 channels
        for channel_num, config in CHANNEL_CONFIG.items():
            if config["display"]:
                print(f"Setting up Channel {channel_num} ({config['label']})...")
                scope.write(f':CHANnel{channel_num}:DISPlay ON')
                scope.write(f':CHANnel{channel_num}:SCALe 2')
                scope.write(f':CHANnel{channel_num}:OFFSet -6')
        
        print("Oscilloscope initialized (All active channels)")
        return scope
    
    def _init_laser(self):
        """Initialize laser for wavelength control"""
        rm = pyvisa.ResourceManager()
        laser = rm.open_resource(LASER_ADDRESS)
        laser.timeout = 5000
        laser.write_termination = ''
        laser.read_termination = ''
        print("Laser initialized")
        return laser
    
    def set_wavelength(self, wavelength_nm):
        """Set laser wavelength and wait for settling"""
        try:
            self.laser.write(f'LW{wavelength_nm}nm')
            print(f"  Setting wavelength to {wavelength_nm}nm...")
            time.sleep(LASER_SETTLING_TIME)
            
            # Verify wavelength setting
            try:
                status = self.laser.query('LS?')
                print(f"  Wavelength confirmed: {status}")
            except:
                print(f"  Wavelength set to {wavelength_nm}nm (verification failed)")
            return True
        except Exception as e:
            print(f"  ERROR: Wavelength setting failed: {e}")
            return False
    
    def turn_laser_on(self):
        """Turn laser ON and wait for it to be ready"""
        try:
            print("  Turning laser ON...")
            self.laser.write('LE1')
            time.sleep(3)  # Wait for laser to actually turn on
            
            # Verify laser is on by checking status (if possible)
            try:
                # Some lasers respond to status queries - adjust command as needed
                status = self.laser.query('LE?')
                print(f"  Laser status: {status}")
            except:
                # If status query fails, just confirm with message
                print("  Laser ON command sent")
            
            # Additional wait for laser to stabilize after turning on
            print(f"  Waiting {LASER_STARTUP_TIME}s for laser to stabilize...")
            time.sleep(LASER_STARTUP_TIME)
            print("  Laser ready")
            return True
        except Exception as e:
            print(f"  ERROR: Laser ON failed: {e}")
            return False
    
    def turn_laser_off(self):
        """Turn laser OFF"""
        try:
            self.laser.write('LE0')
            time.sleep(1)
            print("  Laser OFF")
            return True
        except Exception as e:
            print(f"  ERROR: Laser OFF failed: {e}")
            return False
    
    def send_heater_values(self, config):
        """Send heater voltage configuration to hardware"""
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.serial.write(voltage_message.encode())
        self.serial.flush()
        time.sleep(0.01)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
    
    def measure_all_channels(self):
        """Measure all active channels simultaneously - average of multiple measurements"""
        channel_measurements = {}
        
        # Get list of active channels
        active_channels = [ch for ch, config in CHANNEL_CONFIG.items() if config["display"]]
        
        for channel_num in active_channels:
            measurements = []
            
            for i in range(NUM_MEASUREMENTS_PER_POINT):
                try:
                    value = float(self.scope.query(f':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel{channel_num}'))
                    measurements.append(value)
                    if NUM_MEASUREMENTS_PER_POINT > 1 and i < NUM_MEASUREMENTS_PER_POINT - 1:
                        time.sleep(0.05)  # Brief delay between measurements
                except Exception as e:
                    print(f"    Channel {channel_num} measurement {i+1} failed: {e}")
                    continue
            
            if measurements:
                avg_value = sum(measurements) / len(measurements)
                channel_measurements[channel_num] = round(avg_value, 5)
            else:
                channel_measurements[channel_num] = None
        
        return channel_measurements
    
    def measure_single_wavelength(self, wavelength):
        """Measure all input combinations at a single wavelength for all channels"""
        print(f"\nMeasuring at {wavelength}nm:")
        
        # Set wavelength first
        if not self.set_wavelength(wavelength):
            return None
        
        # Turn on laser and wait for full startup
        if not self.turn_laser_on():
            return None
        
        # Additional settling time after laser is on and wavelength is set
        print(f"  Final settling for {FINAL_SETTLING_TIME}s before measurements...")
        time.sleep(FINAL_SETTLING_TIME)  # Extra time for everything to be completely stable
        
        wavelength_results = {
            'wavelength': wavelength,
            'timestamp': datetime.now().isoformat(),
            'measurements': {}
        }
        
        # Measure each input combination
        for i, input_combo in enumerate(INPUT_COMBINATIONS):
            input_label = INPUT_LABELS[i]
            print(f"  Input {input_label} ({input_combo[0]:.1f}V, {input_combo[1]:.1f}V):")
            
            # Create configuration for this input combination
            current_config = self.base_config.copy()
            current_config[INPUT_HEATERS[0]] = input_combo[0]  # Input A
            current_config[INPUT_HEATERS[1]] = input_combo[1]  # Input B
            
            # Send configuration and wait for settling
            self.send_heater_values(current_config)
            time.sleep(HEATER_SETTLING_TIME)
            
            # Measure all channels
            channel_outputs = self.measure_all_channels()
            
            # Display results for this input combination
            for channel_num, output_value in channel_outputs.items():
                channel_label = CHANNEL_CONFIG[channel_num]["label"]
                if output_value is not None:
                    print(f"    {channel_label}: {output_value:.4f}V")
                else:
                    print(f"    {channel_label}: FAILED")
            
            # Store results
            wavelength_results['measurements'][input_label] = {
                'input_voltages': input_combo,
                'channel_outputs': channel_outputs
            }
        
        self.turn_laser_off()
        return wavelength_results
    
    def run_wavelength_sweep(self):
        """Run the complete wavelength sweep measurement"""
        wavelengths = self.get_wavelength_list()
        
        print(f"\nSTARTING WAVELENGTH SWEEP MEASUREMENT")
        print(f"{'='*60}")
        print(f"Wavelengths to measure: {wavelengths}")
        print(f"Base configuration (non-input heaters):")
        
        # Show base configuration (excluding input heaters)
        base_display = {k: v for k, v in self.base_config.items() 
                       if k not in INPUT_HEATERS and v != 0.01}
        if base_display:
            print(f"  Modified heaters: {base_display}")
        else:
            print(f"  All non-input heaters at 0.01V")
        
        start_time = time.time()
        
        try:
            for i, wavelength in enumerate(wavelengths):
                print(f"\n--- Wavelength {i+1}/{len(wavelengths)} ---")
                
                result = self.measure_single_wavelength(wavelength)
                if result:
                    self.results.append(result)
                else:
                    print(f"  SKIPPED due to errors")
                
                # Show progress
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time_per_wavelength = elapsed / (i + 1)
                    remaining_time = avg_time_per_wavelength * (len(wavelengths) - i - 1)
                    print(f"  Progress: {i+1}/{len(wavelengths)} complete")
                    print(f"  Estimated time remaining: {remaining_time/60:.1f} minutes")
        
        except KeyboardInterrupt:
            print(f"\n\nMeasurement interrupted by user!")
            print(f"Collected data for {len(self.results)} wavelengths before interruption.")
        
        except Exception as e:
            print(f"\n\nUnexpected error: {e}")
            print(f"Collected data for {len(self.results)} wavelengths before error.")
        
        finally:
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"MEASUREMENT COMPLETE")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Wavelengths measured: {len(self.results)}/{len(wavelengths)}")
            
            if self.results:
                self.save_results_clean_format()
                self.print_summary()
    
    def save_results_clean_format(self):
        """Save results in clean format for all channels - fixed Excel formatting"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Get wavelength list and active channels
            wavelengths = self.get_wavelength_list()
            active_channels = [ch for ch, config in CHANNEL_CONFIG.items() if config["display"]]
            
            # Create separate sheet for each channel
            excel_filename = f"wavelength_sweep_4ch_{timestamp}.xlsx"
            
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                for channel_num in active_channels:
                    channel_label = CHANNEL_CONFIG[channel_num]["label"]
                    
                    # Create clean data structure for this channel
                    clean_data = []
                    
                    # Create one row for each input combination
                    for input_label in INPUT_LABELS:
                        input_combo = INPUT_COMBINATIONS[INPUT_LABELS.index(input_label)]
                        
                        # Create row data
                        row_data = {
                            'Gate': GATE_TYPE_MAPPING.get(input_label, f"Gate_{input_label}"),
                            'A': input_combo[0],
                            'B': input_combo[1]
                        }
                        
                        # Add wavelength measurements as columns
                        for wavelength in wavelengths:
                            # Find measurement for this wavelength and input combination
                            measurement_value = None
                            for result in self.results:
                                if result['wavelength'] == wavelength:
                                    if (input_label in result['measurements'] and 
                                        result['measurements'][input_label]['channel_outputs'] and
                                        channel_num in result['measurements'][input_label]['channel_outputs'] and
                                        result['measurements'][input_label]['channel_outputs'][channel_num] is not None):
                                        measurement_value = round(result['measurements'][input_label]['channel_outputs'][channel_num], 3)
                                    break
                            
                            # Use wavelength as column name
                            row_data[str(wavelength)] = measurement_value
                        
                        clean_data.append(row_data)
                    
                    # Create DataFrame for this channel
                    clean_df = pd.DataFrame(clean_data)
                    
                    # Save to Excel sheet
                    sheet_name = f"Ch{channel_num}_{channel_label}"[:31]  # Excel sheet name limit
                    clean_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"4-Channel Excel results saved to: {excel_filename}")
            
            # Also save individual CSV files for each channel
            for channel_num in active_channels:
                channel_label = CHANNEL_CONFIG[channel_num]["label"]
                
                # Create clean data structure for this channel
                clean_data = []
                
                for input_label in INPUT_LABELS:
                    input_combo = INPUT_COMBINATIONS[INPUT_LABELS.index(input_label)]
                    
                    row_data = {
                        'Gate': GATE_TYPE_MAPPING.get(input_label, f"Gate_{input_label}"),
                        'A': input_combo[0],
                        'B': input_combo[1]
                    }
                    
                    for wavelength in wavelengths:
                        measurement_value = None
                        for result in self.results:
                            if result['wavelength'] == wavelength:
                                if (input_label in result['measurements'] and 
                                    result['measurements'][input_label]['channel_outputs'] and
                                    channel_num in result['measurements'][input_label]['channel_outputs'] and
                                    result['measurements'][input_label]['channel_outputs'][channel_num] is not None):
                                    measurement_value = round(result['measurements'][input_label]['channel_outputs'][channel_num], 3)
                                break
                        
                        row_data[str(wavelength)] = measurement_value
                    
                    clean_data.append(row_data)
                
                # Save CSV for this channel
                clean_df = pd.DataFrame(clean_data)
                csv_filename = f"wavelength_sweep_ch{channel_num}_{channel_label}_{timestamp}.csv"
                clean_df.to_csv(csv_filename, index=False)
                
            print(f"Individual CSV files saved for each channel")
            
            # Print summary table for first channel as preview
            if active_channels:
                first_channel = active_channels[0]
                print(f"\n{'='*80}")
                print(f"PREVIEW - Channel {first_channel} ({CHANNEL_CONFIG[first_channel]['label']}) Results:")
                print(f"{'='*80}")
                
                # Create preview data
                preview_data = []
                for input_label in INPUT_LABELS:
                    input_combo = INPUT_COMBINATIONS[INPUT_LABELS.index(input_label)]
                    row_data = {
                        'Gate': GATE_TYPE_MAPPING.get(input_label, f"Gate_{input_label}"),
                        'A': input_combo[0],
                        'B': input_combo[1]
                    }
                    
                    for wavelength in wavelengths[:6]:  # Show first 6 wavelengths
                        measurement_value = None
                        for result in self.results:
                            if result['wavelength'] == wavelength:
                                if (input_label in result['measurements'] and 
                                    result['measurements'][input_label]['channel_outputs'] and
                                    first_channel in result['measurements'][input_label]['channel_outputs'] and
                                    result['measurements'][input_label]['channel_outputs'][first_channel] is not None):
                                    measurement_value = round(result['measurements'][input_label]['channel_outputs'][first_channel], 3)
                                break
                        
                        row_data[str(wavelength)] = measurement_value
                    
                    preview_data.append(row_data)
                
                preview_df = pd.DataFrame(preview_data)
                print(preview_df.to_string(index=False))
                if len(wavelengths) > 6:
                    print(f"... (showing first 6 wavelengths, total: {len(wavelengths)} wavelengths)")
            
        except Exception as e:
            print(f"Error saving clean format results: {e}")
    
    def print_summary(self):
        """Print a summary of the measurement results for all channels"""
        if not self.results:
            print("No results to summarize.")
            return
        
        active_channels = [ch for ch, config in CHANNEL_CONFIG.items() if config["display"]]
        
        print(f"\n{'='*80}")
        print(f"MEASUREMENT SUMMARY - ALL CHANNELS")
        print(f"{'='*80}")
        
        for channel_num in active_channels:
            channel_label = CHANNEL_CONFIG[channel_num]["label"]
            print(f"\n--- Channel {channel_num}: {channel_label} ---")
            
            # Create summary table for this channel
            print(f"{'Wavelength':<12} {'00 (V)':<8} {'01 (V)':<8} {'10 (V)':<8} {'11 (V)':<8}")
            print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            
            for result in self.results:
                row = f"{result['wavelength']:<12.1f}"
                
                for label in INPUT_LABELS:
                    if (label in result['measurements'] and 
                        result['measurements'][label]['channel_outputs'] and
                        channel_num in result['measurements'][label]['channel_outputs'] and
                        result['measurements'][label]['channel_outputs'][channel_num] is not None):
                        voltage = result['measurements'][label]['channel_outputs'][channel_num]
                        row += f" {voltage:<7.3f}"
                    else:
                        row += f" {'FAIL':<7}"
                
                print(row)
            
            # Find interesting patterns for this channel
            print(f"\nAnalysis notes for {channel_label}:")
            
            for result in self.results:
                outputs = []
                for label in INPUT_LABELS:
                    if (label in result['measurements'] and 
                        result['measurements'][label]['channel_outputs'] and
                        channel_num in result['measurements'][label]['channel_outputs'] and
                        result['measurements'][label]['channel_outputs'][channel_num] is not None):
                        outputs.append(result['measurements'][label]['channel_outputs'][channel_num])
                
                if len(outputs) >= 2:
                    output_range = max(outputs) - min(outputs)
                    if output_range > 1.0:  # Significant variation
                        print(f"  λ={result['wavelength']}nm: Large output variation ({output_range:.2f}V range)")
    
    def cleanup(self):
        """Clean up hardware connections"""
        print("\nCleaning up connections...")
        
        try:
            if hasattr(self, 'laser') and self.laser:
                self.laser.write('LE0')  # Turn off laser
                time.sleep(0.5)
                self.laser.close()
                print("  Laser turned OFF and connection closed")
        except Exception as e:
            print(f"  Warning: Laser cleanup failed: {e}")
        
        try:
            if hasattr(self, 'serial') and self.serial:
                self.serial.close()
                print("  Serial connection closed")
        except Exception as e:
            print(f"  Warning: Serial cleanup failed: {e}")
        
        try:
            if hasattr(self, 'scope') and self.scope:
                self.scope.close()
                print("  Scope connection closed")
        except Exception as e:
            print(f"  Warning: Scope cleanup failed: {e}")
        
        print("Cleanup complete")

def main():
    """Main function to run the wavelength sweep measurement"""
    
    # You can specify a custom base configuration here
    # Example with some heaters set to specific values:
    custom_base_config ={0: 0.46, 1: 0.853, 2: 2.151, 3: 2.664, 4: 0.1, 5: 4.04, 6: 0.175, 7: 4.9, 8: 3.563, 9: 4.181, 10: 1.764, 11: 3.06, 12: 0.157, 13: 2.645, 14: 3.851, 15: 2.447, 16: 0.761, 17: 3.616, 18: 0.882, 19: 3.633, 20: 2.76, 21: 4.062, 22: 4.9, 23: 1.255, 24: 2.851, 25: 4.37, 26: 1.059, 27: 2.525, 28: 2.615, 29: 3.38, 30: 2.272, 31: 3.022, 32: 3.678, 33: 0.01, 34: 0.01, 35: 0.01, 36: 0.0, 37: 0.0, 38: 0.01, 39: 0.01}

    # Initialize measurement system
    # Use custom_base_config or None for minimal configuration
    measurement = WavelengthSweepMeasurement(base_config=custom_base_config)
    
    try:
        # Run the complete wavelength sweep
        measurement.run_wavelength_sweep()
        
    except Exception as e:
        print(f"\nUnexpected error in main: {e}")
        
    finally:
        # Always cleanup hardware connections
        measurement.cleanup()

if __name__ == "__main__":
    main()