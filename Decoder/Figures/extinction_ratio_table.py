import pandas as pd
import numpy as np

# Data
# With hill best result
# data_corrected = {
#     "Input_A_New": [0.1, 0.1, 4.9, 4.9],
#     "Input_B_New": [0.1, 4.9, 0.1, 4.9],
#     "Output1": [4.623, 1.986, 3.38, 1.692],
#     "Output2": [1.598, 5.625, 1.273, 3.613],
#     "Output3": [3.513, 3.553, 4.173, 3.368],
#     "Output4": [3.402, 4.911, 3.583, 5.727],
# }

data_corrected = {
    "Input_A_New": [0.1, 0.1, 4.9, 4.9],
    "Input_B_New": [0.1, 4.9, 0.1, 4.9],
    "Output1": [4.995, 3.302, 1.733, 1.71],
    "Output2": [2.462, 4.354, 2.859, 4.468],
    "Output3": [2.897, 3.603, 6.3, 4.033],
    "Output4": [2.132, 3.479, 2.371, 6.01]
}


#With genetic best result



# Create DataFrame
df_corrected = pd.DataFrame(data_corrected)

# Function to calculate normalized values and extinction ratio
def normalize_and_calculate_er(data):
    table_data = []
    for index, row in data.iterrows():
        outputs = [row["Output1"], row["Output2"], row["Output3"], row["Output4"]]
        
        # Identify the dominant and second-highest outputs
        P_max = max(outputs)
        outputs_sorted = sorted(outputs, reverse=True)
        P_second_max = outputs_sorted[1]
        
        # Calculate normalized values relative to P_max
        normalized = [10 * np.log10(P / P_max) if P > 0 else -np.inf for P in outputs]
        
        # Calculate extinction ratio based on dominant and second-highest outputs
        ER = 10 * np.log10(P_max / P_second_max) if P_second_max > 0 else np.inf
        
        table_data.append({
            "Input_A": row["Input_A_New"],
            "Input_B": row["Input_B_New"],
            "Output1": row["Output1"],
            "Output2": row["Output2"],
            "Output3": row["Output3"],
            "Output4": row["Output4"],
            "Norm1": normalized[0],
            "Norm2": normalized[1],
            "Norm3": normalized[2],
            "Norm4": normalized[3],
            "Extinction Ratio (dB)": ER
        })
    return pd.DataFrame(table_data)

# Generate the table
table_df = normalize_and_calculate_er(df_corrected)

# Display the table
print(table_df.to_string(index=False))

# Export the table to LaTeX
with open("table.tex", "w") as f:
    f.write(table_df.to_latex(index=False, float_format="%.2f", caption="Measured Intensities and Extinction Ratios", label="tab:extinction_ratios", longtable=False))

print("LaTeX table has been saved as 'table.tex'")
