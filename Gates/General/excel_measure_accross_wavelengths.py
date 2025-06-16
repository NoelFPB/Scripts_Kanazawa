import serial
import time
import pyvisa
import numpy as np
import pandas as pd
from datetime import datetime
import csv

# Serial port configuration 
SERIAL_PORT = 'COM4'
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
        
        print(f"Wavelength sweep initialized:")
        print(f"  Range: {WAVELENGTH_START}nm to {WAVELENGTH_END}nm")
        print(f"  Step size: {WAVELENGTH_STEP}nm")
        print(f"  Total wavelengths: {len(self.get_wavelength_list())}")
        print(f"  Input combinations: {len(INPUT_COMBINATIONS)}")
        print(f"  Total measurements: {len(self.get_wavelength_list()) * len(INPUT_COMBINATIONS)}")
    
    def get_wavelength_list(self):
        """Generate list of wavelengths to measure"""
        wavelengths = []
        current = WAVELENGTH_START
        while current <= WAVELENGTH_END + 1e-6:  # Small epsilon for floating point comparison
            wavelengths.append(round(current, 1))
            current += WAVELENGTH_STEP
        return wavelengths
    
    def _init_scope(self):
        """Initialize oscilloscope"""
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        if not resources:
            raise Exception("No scope VISA resources found")
        scope = rm.open_resource(resources[0])
        scope.timeout = 5000
        
        # Setup Channel 2 for logic gate output measurement
        scope.write(':CHANnel2:DISPlay ON')
        scope.write(':CHANnel2:SCALe 2')
        scope.write(':CHANnel2:OFFSet -6')
        
        print("Oscilloscope initialized (Channel 2)")
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
    
    def measure_output(self):
        """Measure the logic gate output voltage from oscilloscope - average of multiple measurements"""
        measurements = []
        
        for i in range(NUM_MEASUREMENTS_PER_POINT):
            try:
                value = float(self.scope.query(':MEASure:STATistic:ITEM? CURRent,VMAX,CHANnel2'))
                measurements.append(value)
                if NUM_MEASUREMENTS_PER_POINT > 1 and i < NUM_MEASUREMENTS_PER_POINT - 1:
                    time.sleep(0.1)  # Brief delay between measurements
            except Exception as e:
                print(f"    Measurement {i+1} failed: {e}")
                continue
        
        if measurements:
            avg_value = sum(measurements) / len(measurements)
            return round(avg_value, 5)
        else:
            return None
    
    def measure_single_wavelength(self, wavelength):
        """Measure all input combinations at a single wavelength"""
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
            print(f"  Input {input_label} ({input_combo[0]:.1f}V, {input_combo[1]:.1f}V):", end=" ")
            
            # Create configuration for this input combination
            current_config = self.base_config.copy()
            current_config[INPUT_HEATERS[0]] = input_combo[0]  # Input A
            current_config[INPUT_HEATERS[1]] = input_combo[1]  # Input B
            
            # Send configuration and wait for settling
            self.send_heater_values(current_config)
            time.sleep(HEATER_SETTLING_TIME)
            
            # Measure output - simplified, single measurement
            output_value = self.measure_output()
            
            if output_value is not None:
                print(f"{output_value:.4f}V")
                wavelength_results['measurements'][input_label] = {
                    'input_voltages': input_combo,
                    'output_voltage': output_value
                }
            else:
                print("FAILED")
                wavelength_results['measurements'][input_label] = {
                    'input_voltages': input_combo,
                    'output_voltage': None
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
        """Save results in clean format - fixed Excel formatting"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Get wavelength list for column headers
            wavelengths = self.get_wavelength_list()
            
            # Create clean data structure
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
                                result['measurements'][input_label]['output_voltage'] is not None):
                                measurement_value = round(result['measurements'][input_label]['output_voltage'], 3)
                            break
                    
                    # Use wavelength as column name
                    row_data[str(wavelength)] = measurement_value
                
                clean_data.append(row_data)
            
            # Create DataFrame
            clean_df = pd.DataFrame(clean_data)
            
            # Try to save Excel with simplified formatting
            try:
                excel_filename = f"wavelength_sweep_clean_{timestamp}.xlsx"
                with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                    clean_df.to_excel(writer, sheet_name='Clean_Results', index=False)
                
                print(f"Clean Excel results saved to: {excel_filename}")
                
            except Exception as excel_error:
                print(f"Excel export failed: {excel_error}, saving as CSV instead...")
            
            # Always save CSV (as primary or backup)
            csv_filename = f"wavelength_sweep_clean_{timestamp}.csv"
            clean_df.to_csv(csv_filename, index=False)
            print(f"Clean CSV results saved to: {csv_filename}")
            
            # Print copy-paste ready format
            print(f"\n{'='*80}")
            print(f"COPY-PASTE READY FORMAT (Tab-separated for Excel):")
            print(f"{'='*80}")
            
            # Create tab-separated format
            headers = ["Gate", "A", "B"] + [str(wl) for wl in wavelengths]
            print("\t".join(headers))
            
            for _, row in clean_df.iterrows():
                values = []
                for col in headers:
                    val = row[col]
                    if pd.isna(val) or val is None:
                        values.append("")
                    else:
                        values.append(str(val))
                print("\t".join(values))
            
            print(f"\n{'='*80}")
            print("Copy the lines above and paste directly into Excel!")
            print(f"{'='*80}")
            
            # Print preview of clean data
            print(f"\nClean data preview:")
            print(f"{'='*80}")
            print(clean_df.to_string(index=False, max_cols=8))
            if len(clean_df.columns) > 8:
                print(f"... (showing first 8 columns, total: {len(clean_df.columns)} columns)")
            
        except Exception as e:
            print(f"Error saving clean format results: {e}")
    
    def print_summary(self):
        """Print a summary of the measurement results"""
        if not self.results:
            print("No results to summarize.")
            return
        
        print(f"\n{'='*60}")
        print(f"MEASUREMENT SUMMARY")
        print(f"{'='*60}")
        
        # Create summary table
        print(f"{'Wavelength':<12} {'00 (V)':<8} {'01 (V)':<8} {'10 (V)':<8} {'11 (V)':<8}")
        print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for result in self.results:
            row = f"{result['wavelength']:<12.1f}"
            
            for label in INPUT_LABELS:
                if (label in result['measurements'] and 
                    result['measurements'][label]['output_voltage'] is not None):
                    voltage = result['measurements'][label]['output_voltage']
                    row += f" {voltage:<7.3f}"
                else:
                    row += f" {'FAIL':<7}"
            
            print(row)
        
        # Find interesting patterns
        print(f"\nAnalysis notes:")
        
        # Check for wavelengths with large variations
        for result in self.results:
            outputs = []
            for label in INPUT_LABELS:
                if (label in result['measurements'] and 
                    result['measurements'][label]['output_voltage'] is not None):
                    outputs.append(result['measurements'][label]['output_voltage'])
            
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
    custom_base_config = {0: 1.515, 1: 3.542, 2: 3.549, 3: 0.638, 4: 2.443, 5: 0.1, 6: 1.516, 7: 2.894, 8: 0.1, 9: 0.863, 10: 1.506, 11: 3.42, 12: 0.421, 13: 4.355, 14: 2.68, 15: 0.247, 16: 1.042, 17: 2.08, 18: 3.751, 19: 2.717, 20: 1.99, 21: 0.792, 22: 3.216, 23: 2.804, 24: 0.19, 25: 4.9, 26: 0.147, 27: 0.835, 28: 0.1, 29: 4.449, 30: 4.663, 31: 1.258, 32: 2.184, 33: 0.01, 34: 0.01, 35: 0.01, 36: 0.0, 37: 0.0, 38: 0.01, 39: 0.01} 

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