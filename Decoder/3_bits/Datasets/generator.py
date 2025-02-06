import numpy as np
import pandas as pd

# Define the number of samples per output class
num_samples_per_class = 50

# Generate structured binary-like values for A, B, C with slight noise
data = []
for output_class in range(7):
    for _ in range(num_samples_per_class):
        # Generate base binary values
        binary_values = [int(x) for x in format(output_class, '03b')]  # Convert to 3-bit binary
        
        # Add slight noise
        noisy_values = [round(val + np.random.uniform(-0.1, 0.1),3) for val in binary_values]
        
        # Ensure values remain within [0, 1] bounds
        noisy_values = [max(0.01, min(0.99, v)) for v in noisy_values]
        
        data.append([len(data) + 1] + noisy_values + [f"Out{output_class}"])

# Convert to DataFrame
columns = ["Id", "A", "B", "C", "Output"]
better_df = pd.DataFrame(data, columns=columns)

better_df.to_csv("improved_dataset.csv", index=False, float_format="%.3f")
print(better_df)
