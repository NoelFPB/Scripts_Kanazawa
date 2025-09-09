import pandas as pd
import matplotlib.pyplot as plt
import math

# Load data
df = pd.read_excel('Centered logic_gates_results.xlsx')

# Remap function for clean labels
def remap_value(v):
    if abs(v - 0.1) < 1e-6:
        return 0
    elif abs(v - 4.9) < 1e-6:
        return 1
    else:
        return round(v, 2)

# Generate compact label like (0,1)
df['Label'] = df.apply(lambda row: f"({remap_value(row['A'])},{remap_value(row['B'])})", axis=1)

# Grid size: 3 cols, auto-adjust rows
unique_gates = df['Gate'].unique()
n = len(unique_gates)
cols = 3
rows = math.ceil(n / cols)

# Smaller figure height per row
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3))
axes = axes.flatten()

# Plot
for i, gate in enumerate(unique_gates):
    gate_df = df[df['Gate'] == gate]
    bars = axes[i].bar(range(len(gate_df)), gate_df['Output'], color='skyblue')

    axes[i].set_title(f"{gate} Gate", fontsize=14)
    axes[i].set_ylim(0, 1.1)
    axes[i].set_ylabel("Normalized Output", fontsize=11)
    axes[i].set_xlabel("Input", fontsize=11)
    axes[i].tick_params(axis='y', labelsize=9.5)
    axes[i].set_xticks([])  # remove x-axis tick labels

    # Add threshold line
    axes[i].axhline(y=0.5, color='red', linestyle='--', linewidth=1)

    # Add input labels inside at fixed height
    for j, bar in enumerate(bars):
        axes[i].text(
            bar.get_x() + bar.get_width()/2,
            0.6,  # fixed height
            gate_df['Label'].iloc[j],
            ha='center', va='bottom', fontsize=9, color='black'
        )

# Hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Layout
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.subplots_adjust(wspace=0.4)  # increase horizontal spacing
plt.savefig("logic_gates_subplot_grid.png", dpi=300)
plt.show()
