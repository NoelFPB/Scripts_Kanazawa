import pyvisa
import time

def capture_screenshot(scope: pyvisa.Resource, filename: str = None) -> bool:
    """
    Capture screenshot from RIGOL scope using display data command
    """
    try:
        if filename is None:
            filename = f"scope_capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
            
        print(f"Attempting to save screenshot as {filename}")
        
        # Set longer timeout for image transfer
        scope.timeout = 30000
        
        # Capture the display data
        scope.write(':DISPLAY:DATA? PNG')
        image_data = scope.read_raw()
        
        # Look for PNG signature
        png_start = image_data.find(b'\x89PNG')
        if png_start >= 0:
            # Save the PNG data
            with open(filename, 'wb') as f:
                f.write(image_data[png_start:])
            print(f"Success! Screenshot saved as {filename}")
            return True
        else:
            print("No valid PNG data found")
            return False
            
    except Exception as e:
        print(f"Screenshot capture failed: {e}")
        return False

def main():
    try:
        # Initialize VISA resource manager
        rm = pyvisa.ResourceManager()
        
        # List available resources
        resources = rm.list_resources()
        print("Available resources:", resources)
        
        if not resources:
            raise Exception("No VISA resources found")
        
        # Connect to the first available resource
        scope = rm.open_resource(resources[0])
        
        # Verify connection
        idn = scope.query('*IDN?')
        print(f"Connected to: {idn}")
        
        # Capture the screenshot
        capture_screenshot(scope)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'scope' in locals():
            scope.close()
        if 'rm' in locals():
            rm.close()

if __name__ == "__main__":
    main()