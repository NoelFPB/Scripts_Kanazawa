from utilities import *

def main():
    oscilloscope = OscilloscopeController()  # Using your existing OscilloscopeController class
    config_manager = ConfigurationManager()
    data_processor = DataProcessor(csv_path='C:\\Users\\noelp\\Documents\\Kanazawa\\Scripts_Kanazawa\\MSE\\iris2.csv' )

    serial_controller = SerialController()

    # Setup connections
    if not oscilloscope.connect_oscilloscope():
        print("Failed to connect to oscilloscope")
        return
    if not serial_controller.connect():
        print("Failed to connect to serial port")
        return

    # Load data and create train/test split
    data_processor.load_data()
    
    # Run hill climbing optimization
    best_config, best_mse = serial_controller.hill_climbing(
        config_manager=config_manager,
        data_processor=data_processor,
        oscilloscope=oscilloscope,
        iterations=20
    )

    print(best_config)
    print(best_mse)
    
    # Testing phase
    serial_controller.test(
        data_processor=data_processor,
        oscilloscope=oscilloscope,
        config_manager=config_manager
    )

if __name__ == "__main__":
    main()