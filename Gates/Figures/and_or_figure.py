import matplotlib.pyplot as plt
import numpy as np

# Data from your table (same as before)
input_combinations = ['0-0', '0-1', '1-0', '1-1']
and_output_V = np.array([1.12, 1.02, 1.08, 3.82])
or_output_V = np.array([0.49, 2.14, 2.67, 2.69])

# Normalize the data (same as before)
normalized_and_output = and_output_V / np.max(and_output_V)
normalized_or_output = or_output_V / np.max(or_output_V)

# X-axis labels to match the provided image format
x_labels = [
    "A=0.10\nB=0.10",  # Corresponds to 0-0
    "A=0.10\nB=4.90",  # Corresponds to 0-1
    "A=4.90\nB=0.10",  # Corresponds to 1-0
    "A=4.90\nB=4.90"   # Corresponds to 1-1
]

# Create the 1x2 subplot figure (one row, two columns)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6)) # Adjust figsize as needed for 1x2 layout
# axes is already a 1D array of subplots since nrows=1

# Plot for AND Gate
ax_and = axes[0]
ax_and.bar(x_labels, normalized_and_output, color='skyblue')
ax_and.set_title('AND Gate 1548 nm', fontsize = '20')
ax_and.set_ylabel('Normalized Output', fontsize = '16')
ax_and.set_xlabel('Input', fontsize = '16') # Set x-axis label for this specific subplot
ax_and.axhline(y=0.5, color='r', linestyle='--') # Add red dashed line
ax_and.set_ylim(0, 1.1) # Set y-limit for consistency
ax_and.tick_params(axis='x', labelsize=12) # Adjust x-axis tick label font size

# Plot for OR Gate
ax_or = axes[1]
ax_or.bar(x_labels, normalized_or_output, color='skyblue')
ax_or.set_title('OR Gate 1552 nm', fontsize = '20')
ax_or.set_ylabel('Normalized Output', fontsize = '16')
ax_or.set_xlabel('Input', fontsize = '16') # Set x-axis label for this specific subplot
ax_or.axhline(y=0.5, color='r', linestyle='--') # Add red dashed line
ax_or.set_ylim(0, 1.1) # Set y-limit for consistency
ax_or.tick_params(axis='x', labelsize=12) # Adjust x-axis tick label font size

# Adjust layout to prevent overlap
plt.tight_layout() # Simpler tight_layout for this configuration

# Show the plot
plt.show()