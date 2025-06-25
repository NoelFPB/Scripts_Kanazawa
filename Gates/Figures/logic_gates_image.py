import pandas as pd
import matplotlib.pyplot as plt
import math

# Load data
df = pd.read_excel('Centered logic_gates_results.xlsx')

# Generate label
df['Label'] = df.apply(lambda row: f"A={row['A']:.2f}\nB={row['B']:.2f}", axis=1)

# Grid size: 2 cols, auto-adjust rows
unique_gates = df['Gate'].unique()
n = len(unique_gates)
cols = 3
rows = math.ceil(n / cols)

# Smaller figure height per row
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

# Plot
for i, gate in enumerate(unique_gates):
    gate_df = df[df['Gate'] == gate]
    axes[i].bar(gate_df['Label'], gate_df['Output'], color='skyblue')
    axes[i].set_title(f"{gate} Gate", fontsize=12)
    axes[i].set_ylim(0, 1.1)
    axes[i].set_ylabel("Normalized Output", fontsize=10)
    axes[i].set_xlabel("Input (V)", fontsize=10)
    axes[i].tick_params(axis='x', labelsize=8)
    axes[i].tick_params(axis='y', labelsize=8)
    # Add threshold line
    axes[i].axhline(y=0.5, color='red', linestyle='--', linewidth=1)

# Hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Layout
#plt.suptitle("Photonic Logic Gate Outputs", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("logic_gates_subplot_grid.png", dpi=300)
plt.show()
