# The thing I directly get from the physical decoder should be inputed here.
# Here I check a first condition to test if there is a decoder like behaviour
# I filter and afterwards exclude the configurations columns, for ease of further analysis.


import pandas as pd
import json

# Load the results CSV file
file_path = 'decoder_test_results_with_configurations.csv'
data = pd.read_csv(file_path)

# Convert heater configurations back to dictionaries
data['HeaterConfiguration'] = data['HeaterConfiguration'].apply(json.loads)

# Function to check if outputs obey the decoder behavior
def check_decoder_behavior(df):
    """
    For each input state, checks if exactly one output is at least 1V higher than the others.
    Returns True if this condition holds for all input states.
    """
    for _, row in df.iterrows():
        outputs = [row['Output1'], row['Output2'], row['Output3'], row['Output4']]
        max_output = max(outputs)
        distinct_count = sum(1 for o in outputs if max_output - o >= 1.0)
        # If there isn't exactly one distinct output, return False
        if distinct_count != 3:  # Three outputs must be at least 1V lower
            return False
    return True

# Identify configurations that obey the decoder behavior
decoder_configs = []

for config_num in data['ConfigNumber'].unique():
    # Filter data for the current configuration
    config_data = data[data['ConfigNumber'] == config_num]
    
    # Exclude non-numeric columns before grouping
    numeric_columns = config_data.select_dtypes(include=['number']).columns
    grouped_data = config_data.groupby(['InputState_Heater36', 'InputState_Heater37'], as_index=False)[numeric_columns].mean()

    # Check if the decoder condition holds for all InputStates
    if check_decoder_behavior(grouped_data):
        decoder_configs.append(config_num)

# Display results
if decoder_configs:
    print("Configurations with Decoder-Like Behavior:")
    for config_num in decoder_configs:
        print(f"ConfigNumber: {config_num}")
else:
    print("No configurations with decoder-like behavior were found.")

# Save results to a new CSV file for reference
decoder_results = data[data['ConfigNumber'].isin(decoder_configs)]
columns_to_save = [col for col in decoder_results.columns if col != 'HeaterConfiguration']  # Exclude HeaterConfiguration
decoder_results[columns_to_save].to_csv('decoder_like_configs.csv', index=False)
print("Decoder-like configurations saved to 'decoder_like_configs.csv'.")
