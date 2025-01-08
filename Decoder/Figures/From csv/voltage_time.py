import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv('waveform_20250108_174340.csv')

# Convert time to microseconds
data['Time (µs)'] = data['Time (s)'] * 1e6

# Define colors to match the oscilloscope
scope_colors = {
    'CHANnel1': 'yellow',
    'CHANnel2': 'cyan',
    'CHANnel3': 'magenta',
    'CHANnel4': 'blue'
}

# Create the plot
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')  # Use dark background to match scope

# Plot each channel
for channel in data['Channel'].unique():
    channel_data = data[data['Channel'] == channel]
    plt.plot(channel_data['Time (µs)'], channel_data['Voltage (V)'], 
             color=scope_colors[channel], 
             label=channel,
             linewidth=1.5)

# Set axis limits with some padding
voltage_min = data['Voltage (V)'].min()
voltage_max = data['Voltage (V)'].max()
voltage_range = voltage_max - voltage_min
voltage_padding = voltage_range * 0.1

plt.ylim(voltage_min - voltage_padding, voltage_max + voltage_padding)

# Configure axis ticks and grid
plt.xlabel('Time (µs)')
plt.ylabel('Voltage (V)')
plt.title('Oscilloscope Capture')
plt.grid(True, alpha=0.3, linestyle='--')

# Add more tick marks
plt.minorticks_on()
plt.grid(True, which='minor', alpha=0.1, linestyle=':')

# Format y-axis ticks to show voltage with 2 decimal places
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f V'))

# Format x-axis ticks to show time in µs with 1 decimal place
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f µs'))

# Add legend with semi-transparent background
plt.legend(framealpha=0.7)

# Adjust layout to prevent label clipping
plt.tight_layout()

# Show plot with a dark gray border
plt.gcf().patch.set_facecolor('#2F2F2F')

plt.show()

# Print timing information
time_range = data['Time (µs)'].max() - data['Time (µs)'].min()
print(f"\nCapture Information:")
print(f"Time Range: {time_range:.2f} µs")
print(f"Voltage Range: {voltage_min:.2f}V to {voltage_max:.2f}V")