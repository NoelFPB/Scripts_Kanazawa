import pandas as pd
import numpy as np

def normalize_decoder(input_file='decoder_data.csv', output_file='decoder_normalized.csv'):
    # Read the original dataset
    df = pd.read_csv(input_file)
    
    # Create a new dataframe for normalized values
    df_normalized = pd.DataFrame()
    df_normalized['Id'] = df['Id']
    
    # Define voltage range
    V_MIN = 0.1
    V_MAX = 4.9
    
    # Normalize each input signal
    features = ['A', 'B', 'C']
    
    for feature in features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"{feature} range: {min_val:.2f} to {max_val:.2f}")
        
        # Apply min-max normalization to voltage range
        df_normalized[feature] = V_MIN + (df[feature] - min_val) * (V_MAX - V_MIN) / (max_val - min_val)
        
        # Round to 2 decimal places for cleaner values
        df_normalized[feature] = df_normalized[feature].round(2)
    
    # Keep the output column
    df_normalized['Output'] = df['Output']
    
    # Print normalized ranges for verification
    print("\nNormalized ranges:")
    for feature in features:
        print(f"{feature}: {df_normalized[feature].min():.2f} to {df_normalized[feature].max():.2f}")
    
    # Calculate threshold voltage for digital interpretation
    V_THRESHOLD = (V_MAX + V_MIN) / 2
    print(f"\nVoltage threshold for digital interpretation: {V_THRESHOLD:.2f}V")
    
    # Verify digital interpretation
    digital_values = df_normalized[features].applymap(lambda x: 1 if x > V_THRESHOLD else 0)
    print("\nSample of digital interpretations:")
    print(digital_values.head())
    
    # Save to new CSV file
    df_normalized.to_csv(output_file, index=False)
    print(f"\nNormalized data saved to {output_file}")
    
    # Print sample rows for verification
    print("\nFirst few rows of normalized data:")
    print(df_normalized.head())

if __name__ == "__main__":
    normalize_decoder()