import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# New data format: each row is a case, and 'Outputs' is a list of 4 values
data_corrected = {
    "Input_A": [0.1, 0.1, 4.9, 4.9],
    "Input_B": [0.1, 4.9, 0.1, 4.9],
    "Outputs": [
    [5.36367, 1.972, 1.22733, 1.22633],    # 00 (A=0.10V, B=0.10V)
    [2.32133, 5.13733, 1.62567, 1.84633],  # 01 (A=0.10V, B=4.90V)
    [1.278, 0.60053, 5.40467, 1.95733],    # 10 (A=4.90V, B=0.10V)
    [0.71353, 2.03167, 1.618, 5.57367]     # 11 (A=4.90V, B=4.90V)
]
}

# Create DataFrame
df_corrected = pd.DataFrame(data_corrected)

# Channel colors
channel_colors = ["#FFD700", "#00FFFF", "#FF00FF", "#0080FF"]

# Normalize a list
def normalize_vector(vec):
    max_val = max(vec)
    return [v / max_val for v in vec] if max_val != 0 else [0] * len(vec)

# Extinction ratio calculation
def calculate_extinction_ratio(vec):
    sorted_vals = sorted(vec, reverse=True)
    if sorted_vals[1] == 0:
        return float('inf')
    return 10 * np.log10(sorted_vals[0] / sorted_vals[1])

# Plot and save
def save_normalized_plots(fig_width, fig_height):
    base_fontsize = max(25, min(fig_width, fig_height) * 1.5)

    for i, row in df_corrected.iterrows():
        outputs = row["Outputs"]
        normalized_outputs = normalize_vector(outputs)
        er = calculate_extinction_ratio(normalized_outputs)
        print(f"Case {i+1} (Inputs A={row['Input_A']}, B={row['Input_B']}) - Extinction Ratio: {er:.2f} dB")

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.bar(["0", "1", "2", "3"], normalized_outputs, color=channel_colors, alpha=0.8)
        ax.set_ylabel("Normalized Voltage", fontsize=base_fontsize*0.9)
        ax.tick_params(axis='x', labelsize=base_fontsize * 0.9)
        ax.tick_params(axis='y', labelsize=base_fontsize * 0.9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"normalized_result{i+1}.png", dpi=300)
        plt.close()

# Run
save_normalized_plots(5, 4.07)
