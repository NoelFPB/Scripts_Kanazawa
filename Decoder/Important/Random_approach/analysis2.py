import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('decoder_like_configs.csv')

# Define a function to check decoder-like behavior for a group of rows
def check_unique_dominance(group, threshold=1.0):
    results = []
    dominant_bits = {}
    
    for index, row in group.iterrows():
        # Get the input state and outputs
        input_state = (row['InputState_Heater36'], row['InputState_Heater37'])
        outputs = [row['Output1'], row['Output2'], row['Output3'], row['Output4']]
        max_val = max(outputs)
        # Identify the index of the dominant output
        dominant_index = outputs.index(max_val)
        # Ensure the dominant output is significantly higher than the rest
        differences = [max_val - output for output in outputs]
        distinctly_higher = sum(diff > threshold for diff in differences) == (len(outputs) - 1)
        
        if distinctly_higher:
            # Check if this dominant index is already associated with another input state
            if dominant_index in dominant_bits.values():
                print(f"Conflict found: Input state {input_state} produces the same dominant output as another row.")
                return False  # Conflict found, not decoder-like behavior
            # Otherwise, save the dominant index for this input state
            dominant_bits[input_state] = dominant_index
            results.append((index, input_state, dominant_index))
    
    return results

# Group by ConfigNumber and apply the check function
decoder_like_results = {}
for config_number, group in df.groupby('ConfigNumber'):
    print(f"Analyzing ConfigNumber {config_number}...")
    decoder_indices = check_unique_dominance(group)
    if decoder_indices:  # If no conflicts and all rows meet criteria
        decoder_like_results[config_number] = decoder_indices

# Display the results
for config, results in decoder_like_results.items():
    print(f"ConfigNumber {config} has unique decoder-like behavior:")
    for row_info in results:
        print(f"  Row {row_info[0]}: InputState {row_info[1]} -> Dominant Output Index {row_info[2]}")
