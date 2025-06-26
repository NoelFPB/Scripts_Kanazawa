import pyvisa
import time

def find_laser_device():
    """
    Scan for available VISA resources and find the laser
    """
    print("=== Scanning for VISA Resources ===")
    try:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        
        print(f"Found {len(resources)} VISA resources:")
        for i, resource in enumerate(resources):
            print(f"  {i+1}. {resource}")
        
        if not resources:
            print("❌ No VISA resources found!")
            print("\nTroubleshooting steps:")
            print("1. Check USB cable connection")
            print("2. Verify laser is powered on")
            print("3. Install/reinstall VISA drivers")
            print("4. Try different USB port")
            return None
            
        # Look for likely laser candidates
        laser_candidates = []
        for resource in resources:
            # Common patterns for ALnair or similar devices
            if any(pattern in resource.upper() for pattern in ['USB', 'GPIB', 'TCPIP', '0610', 'ALNAIR']):
                laser_candidates.append(resource)
        
        if laser_candidates:
            print(f"\nPossible laser devices: {laser_candidates}")
            return laser_candidates
        else:
            print("\nNo obvious laser candidates found. All resources listed above.")
            return resources
            
    except Exception as e:
        print(f"❌ Error scanning resources: {e}")
        return None

def test_connection(resource_string):
    """
    Test connection to a specific resource
    """
    print(f"\n--- Testing connection to: {resource_string} ---")
    try:
        rm = pyvisa.ResourceManager()
        device = rm.open_resource(resource_string)
        device.timeout = 5000
        device.write_termination = ''
        device.read_termination = ''
        
        # Try to get device identification
        try:
            device.write('*IDN?')
            time.sleep(0.5)
            response = device.read()
            print(f"✅ Device responded: {response}")
            device.close()
            return True
        except:
            # If *IDN? doesn't work, try ALnair specific commands
            try:
                device.write('LS?')  # Status query
                time.sleep(0.5)
                response = device.read()
                print(f"✅ ALnair device confirmed: {response}")
                device.close()
                return True
            except:
                print("❌ Device doesn't respond to standard queries")
                device.close()
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def auto_find_laser():
    """
    Automatically find and connect to the laser
    """
    print("=== Auto-detecting ALnair TLG220 ===")
    
    # First, scan for resources
    candidates = find_laser_device()
    if not candidates:
        return None
    
    # Test each candidate
    for resource in candidates:
        if test_connection(resource):
            print(f"🎯 Found working laser at: {resource}")
            return resource
    
    print("❌ No working laser connection found")
    return None

def run_laser_sequence(resource_string=None):
    """
    Fully automated laser sequence using confirmed working commands
    """
    
    print("=== ALnair TLG220 Automated Sequence ===")
    
    # Auto-detect if no resource specified
    if resource_string is None:
        resource_string = auto_find_laser()
        if resource_string is None:
            print("❌ Cannot proceed without valid connection")
            return
    
    # Connect to laser
    try:
        rm = pyvisa.ResourceManager()
        laser = rm.open_resource(resource_string)
        laser.timeout = 5000
        laser.write_termination = ''
        laser.read_termination = ''
        print(f"✅ Connected to laser at {resource_string}")
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
        laser.write('LW1552nm')  # Fixed: was 1550nm, should be 1552nm
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
        try:
            laser.write('LE0')  # Safety: ensure laser is off
            print("🛡️ Laser turned OFF for safety")
        except:
            pass
        laser.close()
        print("✅ Disconnected from laser")

def interactive_laser_control(resource_string=None):
    """
    Interactive control using the discovered commands
    """
    
    # Auto-detect if no resource specified
    if resource_string is None:
        resource_string = auto_find_laser()
        if resource_string is None:
            print("❌ Cannot proceed without valid connection")
            return
    
    # Connect to laser
    try:
        rm = pyvisa.ResourceManager()
        laser = rm.open_resource(resource_string)
        laser.timeout = 5000
        laser.write_termination = ''
        laser.read_termination = ''
        print(f"✅ Connected to laser at {resource_string}")
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
                run_laser_sequence(resource_string)
                return
                
            elif choice == '7':
                break
                
            else:
                print("Invalid choice")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        try:
            laser.write('LE0')  # Safety: turn off laser
            print("🛡️ Laser turned OFF for safety")
        except:
            pass
        laser.close()
        print("✅ Disconnected")

def manual_connection_test():
    """
    Manually test connection with user-provided resource string
    """
    print("=== Manual Connection Test ===")
    resource_string = input("Enter VISA resource string (e.g., 'USB0::0x1AB1::0x0610::HDO1B244000779::INSTR'): ").strip()
    
    if test_connection(resource_string):
        print("✅ Connection successful!")
        
        use_device = input("Use this device? (y/n): ").strip().lower()
        if use_device == 'y':
            return resource_string
    else:
        print("❌ Connection failed")
    
    return None

if __name__ == "__main__":
    print("=== ALnair TLG220 Laser Control ===")
    print("Choose mode:")
    print("1. Auto-detect and run sequence")
    print("2. Auto-detect and interactive control")
    print("3. Scan for devices only")
    print("4. Manual connection test")
    print("5. Run sequence with known resource")
    
    choice = input("Enter 1-5: ").strip()
    
    if choice == "1":
        run_laser_sequence()
    elif choice == "2":
        interactive_laser_control()
    elif choice == "3":
        find_laser_device()
    elif choice == "4":
        resource = manual_connection_test()
        if resource:
            mode = input("Run sequence (s) or interactive (i)? ").strip().lower()
            if mode == 's':
                run_laser_sequence(resource)
            elif mode == 'i':
                interactive_laser_control(resource)
    elif choice == "5":
        resource = input("Enter resource string: ").strip()
        run_laser_sequence(resource)
    else:
        print("Invalid choice")