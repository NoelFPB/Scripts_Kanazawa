import pandas as pd
import numpy as np

# normalized_value = V_MIN + (value - min_val) * (V_MAX - V_MIN) / (max_val - min_val)
# V_MIN = 0.1 (minimum voltage)
# V_MAX = 4.9 (maximum voltage)
# value = current measurement (e.g., a sepal length)
# min_val = minimum value in the original feature (minimum e.g sepal lenght among all the samples)
# max_val = maximum value in the original feature 


def normalize_iris(input_file='iris.csv', output_file='iris_normalized.csv'):
    # Read the original dataset
    df = pd.read_csv(input_file)
    
    # Create a new dataframe for normalized values
    df_normalized = pd.DataFrame()
    df_normalized['Id'] = df['Id']
    
    # Define voltage range
    V_MIN = 0.1
    V_MAX = 4.9
    
    # Normalize each feature to the voltage range
    features = {
        'SepalLengthCm': 'SeL',
        'SepalWidthCm': 'SeW',
        'PetalLengthCm': 'PeL',
        'PetalWidthCm': 'PeW'
    }
    
    for original, new in features.items():
        min_val = df[original].min()
        max_val = df[original].max()
        print(min_val)
        print(max_val)
        # Apply min-max normalization to voltage range
        df_normalized[new] = V_MIN + (df[original] - min_val) * (V_MAX - V_MIN) / (max_val - min_val)
        
        # Round to 2 decimal places for cleaner values
        df_normalized[new] = df_normalized[new].round(2)
    
    # Keep the species column
    df_normalized['Species'] = df['Species']
    
    # Print value ranges for verification
    print("Normalized ranges:")
    for col in ['SeL', 'SeW', 'PeL', 'PeW']:
        print(f"{col}: {df_normalized[col].min():.2f} to {df_normalized[col].max():.2f}")
    
    # Save to new CSV file
    df_normalized.to_csv(output_file, index=False)
    print(f"\nNormalized data saved to {output_file}")
    
    # Print sample rows for verification
    print("\nFirst few rows of normalized data:")
    print(df_normalized.head())

if __name__ == "__main__":
    normalize_iris()