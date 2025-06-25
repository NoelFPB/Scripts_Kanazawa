import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
data_corrected = {
    "Input_A": [0.1, 0.1, 4.9, 4.9],
    "Input_B": [0.1, 4.9, 0.1, 4.9],
    "Outputs": [
        [5.58, 2.15, 1.10, 1.46],  # 00
        [1.41, 5.68, 0.56, 0.68],  # 01
        [2.83, 1.08, 5.50, 1.71],  # 10
        [0.66, 2.90, 2.79, 5.61]   # 11
    ]
}

# Create DataFrame
df_corrected = pd.DataFrame(data_corrected)

# Colors
channel_colors = ["#FFD700", "#00FFFF", "#FF00FF", "#0080FF"]

# Normalize
def normalize_vector(vec):
    max_val = max(vec)
    return [v / max_val for v in vec] if max_val != 0 else [0] * len(vec)

# ER
def calculate_extinction_ratio(vec):
    sorted_vals = sorted(vec, reverse=True)
    if sorted_vals[1] == 0:
        return float('inf')
    return 10 * np.log10(sorted_vals[0] / sorted_vals[1])

# Plot# Plot
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(hspace=0.6, wspace=0.3)
axs = axs.flatten()

for i, (idx, row) in enumerate(df_corrected.iterrows()):
    outputs = row["Outputs"]
    normalized_outputs = normalize_vector(outputs)
    er = calculate_extinction_ratio(normalized_outputs)
    ax = axs[i]

    bars = ax.bar(["0", "1", "2", "3"], normalized_outputs, color=channel_colors, alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Output", fontsize=18)
    ax.set_ylabel("Normalized Voltage", fontsize=16)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Clear title (remove ax.set_title) and add custom title above plot
        # Add "Input" above the input values
      # Add invisible placeholders to reserve vertical space
    # Reserve space for titles (invisible text)
    ax.text(
        0.02, 1.15,
        " ",  # Placeholder for "Input"
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=13
    )

    ax.text(
        0.02, 1.05,
        " ",  # Placeholder for input values
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=12
    )

    ax.text(
        0.98, 1.05,
        " ",  # Placeholder for ER value
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=12
    )

    # Annotate the tallest bar with its index inside the bar
    max_index = np.argmax(normalized_outputs)
    ax.text(
        max_index,
        normalized_outputs[max_index] / 2,
        str(max_index),
        ha='center', va='center',
        fontsize=14, fontweight='bold', color='black'
    )



plt.tight_layout()
plt.savefig("poster_grid_plot_annotated.png", dpi=300)
plt.show()
