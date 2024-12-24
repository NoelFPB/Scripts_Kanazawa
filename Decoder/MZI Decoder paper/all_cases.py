import numpy as np
import matplotlib.pyplot as plt

# Define constants
AMPLITUDE_A = 1  # Amplitude of signal A
AMPLITUDE_B = 1  # Amplitude of signal B
AMPLITUDE_ASSIST = 2  # Amplitude of assist signal
PHASES = {
    "0": 0,           # Phase for binary 0
    "1": np.pi        # Phase for binary 1
}

# Generate the wave for a given phase
def generate_wave(amplitude, phase, t):
    return amplitude * np.cos(2 * np.pi * t + phase)

# Time range for plotting (2 periods)
num_points = 1000
t = np.linspace(0, 2, num_points)  # 2 periods

# Function to compute intensity for a binary combination of A and B with phase shifts
def compute_intensity(combo, phase_shift_a=0, phase_shift_b=0):
    phase_a = PHASES[combo[0]] + phase_shift_a  # Signal A phase
    phase_b = PHASES[combo[1]] + phase_shift_b  # Signal B phase
    phase_assist = PHASES["0"]  # Assist signal phase

    # Generate waves
    wave_a = generate_wave(AMPLITUDE_A, phase_a, t)
    wave_b = generate_wave(AMPLITUDE_B, phase_b, t)
    assist_wave = generate_wave(AMPLITUDE_ASSIST, phase_assist, t)

    # Combined wave
    combined_wave = wave_a + wave_b + assist_wave

    # Compute intensity as the square of the amplitude
    instantaneous_power = combined_wave ** 2

    # Average intensity over the entire time range
    average_intensity = np.mean(instantaneous_power) / 2

    return average_intensity

# Define binary combinations of A and B
combinations = ["00", "01", "10", "11"]

# Define phase shift configurations
scenarios = {
    "0-0": (0, 0),
    "0-π": (0, np.pi),
    "π-0": (np.pi, 0),
    "π-π": (np.pi, np.pi)
}

# Compute intensities for each scenario
scenario_intensities = {}
for scenario, (phase_shift_a, phase_shift_b) in scenarios.items():
    intensities = [compute_intensity(combo, phase_shift_a, phase_shift_b) for combo in combinations]
    scenario_intensities[scenario] = intensities

# Plot the results
plt.figure(figsize=(12, 8))
bar_width = 0.2
x = np.arange(len(combinations))

for i, (scenario, intensities) in enumerate(scenario_intensities.items()):
    plt.bar(x + i * bar_width, intensities, width=bar_width, label=scenario)

plt.title("Relative Intensities for Different Phase Scenarios", fontsize=14)
plt.xlabel("Binary Combination of A and B", fontsize=12)
plt.ylabel("Intensity", fontsize=12)
plt.xticks(x + bar_width * 1.5, combinations, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Scenarios", fontsize=10)

# Annotate intensities
for i, (scenario, intensities) in enumerate(scenario_intensities.items()):
    for j, intensity in enumerate(intensities):
        plt.text(x[j] + i * bar_width, intensity + 0.2, f"{intensity:.2f}", ha='center', fontsize=9)

plt.tight_layout()
plt.show()
