import matplotlib.pyplot as plt
import pandas as pd

# Normalized intensity values in dB
data_normalized = {
    "Input_A_New": [0.1, 0.1, 4.9, 4.9],
    "Input_B_New": [0.1, 4.9, 0.1, 4.9],
    "O1": [0.00, -1.05, -4.57, -4.30],
    "O2": [-2.53, 0.00, -3.44, -1.54],
    "O3": [-2.10, -0.75, 0.00, -1.98],
    "O4": [-2.86, -0.88, -3.93, 0.00]
}

# Create DataFrame
df_normalized = pd.DataFrame(data_normalized)

# Define channel colors
channel_colors = ["#FFD700", "#00FFFF", "#FF00FF", "##0080FF"]  # Yellow, Cyan, Magenta, Green

# Function to generate and save individual plots
def save_normalized_plots(fig_width, fig_height):
    base_fontsize = max(25, min(fig_width, fig_height) * 1.5)

    for i in range(len(df_normalized)):
        values = [
            df_normalized.loc[i, "O1"],
            df_normalized.loc[i, "O2"],
            df_normalized.loc[i, "O3"],
            df_normalized.loc[i, "O4"]
        ]
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.bar(["O1", "O2", "O3", "O4"], values, color=channel_colors, alpha=0.8)
        ax.set_ylabel("Normalized Intensity (dB)", fontsize=base_fontsize)
        ax.set_ylim(-6, 1)  # Adjust if needed to see all bars clearly
        ax.tick_params(axis='x', labelrotation=0, labelsize=base_fontsize * 0.9)
        ax.tick_params(axis='y', labelsize=base_fontsize * 0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"normalized_result{i+1}.png", dpi=300)
        plt.close()

# Call the function
save_normalized_plots(5, 4)
