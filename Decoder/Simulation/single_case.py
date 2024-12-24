import numpy as np
import matplotlib.pyplot as plt

# Define constants
AMPLITUDE_A =1  # Amplitude of signal A
AMPLITUDE_B = 1  # Amplitude of signal B
AMPLITUDE_ASSIST = 2  # Amplitude of assist signal
PHASES = {
    "0": 0,           # Phase for binary 0
    "1": np.pi        # Phase for binary 1
}

# Generate the wave for a given phase
def generate_wave(amplitude, phase, t):
    return amplitude * np.sin(2 * np.pi * t + phase)

# Time range for plotting (2 periods)
num_points = 1000
t = np.linspace(0, 2, num_points)  # 2 periods

# Function to compute intensity for a binary combination of A and B
def compute_intensity(combo):
    phase_a = PHASES[combo[0]]  # Signal A phase
    phase_b = PHASES[combo[1]]  # Signal B phase
    phase_assist = PHASES["0"]  # Assist signal phase (0 for the 0-0 scenario)

    # Generate waves
    wave_a = generate_wave(AMPLITUDE_A, phase_a, t)
    wave_b = generate_wave(AMPLITUDE_B, phase_b, t)
    assist_wave = generate_wave(AMPLITUDE_ASSIST, phase_assist, t)



    # Combined wave
    combined_wave = wave_a + wave_b + assist_wave


    
    # Compute intensity as the square of the amplitude
    instantaneous_power = combined_wave ** 2
    # Average intensity over the entire time range
    average_intensity = np.mean(instantaneous_power)/2

    # plot wave
    plt.plot(t, wave_a, label = 'wave A')
    plt.plot(t, wave_b, label = 'wave B')
    plt.plot(t, wave_a + wave_b, label = 'wavea + wave B')
    
    plt.plot(t, assist_wave, label = 'assist wave')
    plt.plot(t,combined_wave/2, label = 'target wave')
    plt.plot(t, instantaneous_power/4, label = 'instantaneous power')

    plt.legend()
    plt.show()

    print(average_intensity)
    return average_intensity

# Define binary combinations of A and B
combinations = ["00", "01", "10", "11"]

# Compute intensities for each combination
intensities = [compute_intensity(combo) for combo in combinations]


# Plot the results
plt.figure(figsize=(8, 6))
plt.bar(combinations, intensities, color='skyblue', edgecolor='black')
plt.title("Relative Intensities for 0-0 Scenario", fontsize=14)
plt.xlabel("Binary Combination of A and B", fontsize=12)
plt.ylabel("Intensity", fontsize=12)
plt.ylim(0, max(intensities) + 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Annotate the intensities on the bars
for i, intensity in enumerate(intensities):
    plt.text(i, intensity + 0.2, f"{intensity:.2f}", ha='center', fontsize=10)

plt.show()
