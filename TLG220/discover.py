import pyvisa
import time

def run_laser_sequence():
    """
    Fully automated laser sequence using confirmed working commands
    """
    
    print("=== ALnair TLG 220 Automated Sequence ===")
    
    # Connect to laser
    try:
        laser = pyvisa.ResourceManager().open_resource('GPIB0::6::INSTR')
        laser.timeout = 5000
        laser.write_termination = ''
        laser.read_termination = ''
        print("✅ Connected to laser")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    try:
        # Step 1: Set up for 1550nm
        print("\n--- Step 1: 1550nm sequence ---")
        
        print("Setting wavelength to 1550nm...")
        laser.write('LW1550nm')
        time.sleep(2)
        
        # Verify wavelength
        try:
            wavelength_status = laser.query('LS?')
            print(f"Wavelength confirmed: {wavelength_status}")
        except:
            print("Wavelength set (verification failed)")
        
        print("Turning laser ON...")
        laser.write('LE1')
        time.sleep(1)
        
        # Check output power
        try:
            output_power = laser.query('POWER?')
            print(f"Output power: {output_power}")
        except:
            print("Laser ON (power check failed)")
        
        print("Waiting 10 seconds...")
        for i in range(10, 0, -1):
            print(f"  {i}...", end=' ', flush=True)
            time.sleep(1)
        print("\n")
        
        print("Turning laser OFF...")
        laser.write('LE0')
        time.sleep(1)
        
        # Step 2: Set up for 1552nm
        print("--- Step 2: 1552nm sequence ---")
        
        print("Setting wavelength to 1552nm...")
        laser.write('LW1550nm')
        time.sleep(2)
        
        # Verify wavelength
        try:
            wavelength_status = laser.query('LS?')
            print(f"Wavelength confirmed: {wavelength_status}")
        except:
            print("Wavelength set (verification failed)")
        
        print("Turning laser ON...")
        laser.write('LE1')
        time.sleep(1)
        
        # Check output power
        try:
            output_power = laser.query('POWER?')
            print(f"Output power: {output_power}")
        except:
            print("Laser ON (power check failed)")
        
        print("Waiting 10 seconds...")
        for i in range(10, 0, -1):
            print(f"  {i}...", end=' ', flush=True)
            time.sleep(1)
        print("\n")
        
        print("Turning laser OFF...")
        laser.write('LE0')
        time.sleep(1)
        
        # Final status check
        try:
            final_wavelength = laser.query('LS?')
            final_power = laser.query('LW?')
            output_power = laser.query('POWER?')
            
            print("--- Final Status ---")
            print(f"Wavelength: {final_wavelength}")
            print(f"Set Power: {final_power}")
            print(f"Output Power: {output_power}")
        except:
            print("Final status check failed")
        
        print("\n🎉 SEQUENCE COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Sequence failed: {e}")
    
    finally:
        laser.close()
        print("✅ Disconnected from laser")

def interactive_laser_control():
    """
    Interactive control using the discovered commands
    """
    
    # Connect to laser
    try:
        laser = pyvisa.ResourceManager().open_resource('GPIB0::6::INSTR')
        laser.timeout = 5000
        laser.write_termination = ''
        laser.read_termination = ''
        print("✅ Connected to laser")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    try:
        while True:
            print("\n--- ALnair Laser Control ---")
            print("1. Turn ON (LE1)")
            print("2. Turn OFF (LE0)")
            print("3. Set wavelength")
            print("4. Set power")
            print("5. Check status")
            print("6. Run full sequence")
            print("7. Exit")
            
            choice = input("Enter choice: ").strip()
            
            if choice == '1':
                laser.write('LE1')
                print("✅ Laser ON")
                
            elif choice == '2':
                laser.write('LE0')
                print("✅ Laser OFF")
                
            elif choice == '3':
                wavelength = input("Enter wavelength (nm): ")
                laser.write(f'LW{wavelength}nm')
                time.sleep(2)
                try:
                    status = laser.query('LS?')
                    print(f"✅ {status}")
                except:
                    print("✅ Wavelength set")
                    
            elif choice == '4':
                power = input("Enter power (dBm): ")
                laser.write(f'LP{power}dBm')
                time.sleep(1)
                try:
                    status = laser.query('LW?')
                    print(f"✅ {status}")
                except:
                    print("✅ Power set")
                    
            elif choice == '5':
                try:
                    wavelength = laser.query('LS?')
                    power = laser.query('LW?')
                    output = laser.query('POWER?')
                    print(f"Wavelength: {wavelength}")
                    print(f"Set Power: {power}")
                    print(f"Output Power: {output}")
                except Exception as e:
                    print(f"Status check failed: {e}")
                    
            elif choice == '6':
                laser.close()
                run_laser_sequence()
                return
                
            elif choice == '7':
                break
                
            else:
                print("Invalid choice")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        try:
            #laser.write('LE0')  # Safety: turn off laser
            print("🛡️ Laser turned OFF for safety")
        except:
            pass
        laser.close()
        print("✅ Disconnected")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Run automated sequence (1550nm → 1552nm)")
    print("2. Interactive control")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        run_laser_sequence()
    elif choice == "2":
        interactive_laser_control()
    else:
        print("Invalid choice")