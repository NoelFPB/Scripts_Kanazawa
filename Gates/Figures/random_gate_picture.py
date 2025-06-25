import pandas as pd
import matplotlib.pyplot as plt

# ========================================
# MANUAL INPUT SECTION - EDIT HERE
# ========================================

# Gate type
GATE_TYPE = "XOR"  # Change this to your desired gate

# Input values (A, B) and corresponding outputs
# Format: [(A_value, B_value, Output_value), ...]
GATE_DATA = [
    (0.1, 0.1,  0.24),   # A=0.1, B=0.1 -> Output=0.85
    (0.1, 4.9, 0.95),   # A=0.1, B=4.9 -> Output=0.12
    (4.9, 0.1, 1),   # A=4.9, B=0.1 -> Output=0.08
    (4.9, 4.9, 0.09),   # A=4.9, B=4.9 -> Output=0.78
]

# Plot customization
FIGURE_SIZE = (8, 6)  # (width, height) in inches
DPI = 300
SAVE_FILENAME = f"{GATE_TYPE}_gate_output.png"
THRESHOLD_VALUE = 0.5

# ========================================
# PLOTTING CODE - NO NEED TO EDIT
# ========================================

# Create DataFrame from manual data
df = pd.DataFrame(GATE_DATA, columns=['A', 'B', 'Output'])

# Generate labels for x-axis
df['Label'] = df.apply(lambda row: f"A={row['A']:.1f}\nB={row['B']:.1f}", axis=1)

# Create the plot
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

# Create bar plot
bars = ax.bar(df['Label'], df['Output'], color='skyblue')

# Customize the plot
ax.set_title(f"{GATE_TYPE}", fontsize=20, fontweight='bold', pad=20)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Normalized Output", fontsize=20)
ax.set_xlabel("Input Combination", fontsize=20)

# Improve tick labels
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig(SAVE_FILENAME, dpi=DPI, bbox_inches='tight')

# Show the plot
plt.show()

print(f"Plot saved as: {SAVE_FILENAME}")
print(f"Gate: {GATE_TYPE}")
print("Input-Output mapping:")
for a, b, output in GATE_DATA:
    print(f"  A={a:.1f}, B={b:.1f} -> Output={output:.3f}")