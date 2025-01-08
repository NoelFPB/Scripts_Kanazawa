import numpy as np
import matplotlib.pyplot as plt

# Define constants
AMPLITUDE_A = 1  # Amplitude of signal A
AMPLITUDE_B = 1  # Amplitude of signal B
AMPLITUDE_ASSIST = 1  # Amplitude of assist signal
PHASES = {
    "0": 0,           # Phase for binary 0
    "1": np.pi        # Phase for binary 1
}

# Generate the wave for a given phase
def generate_wave(amplitude, phase, t):
    return amplitude * np.cos(2 * np.pi * t + phase)

# Time range for plotting (2 periods)
num_points = 1000
t = np.linspace(0, 1, num_points)  # 2 periods

# Phases for the 0-0 scenario
phase_a = PHASES["0"]  # Signal A phase
phase_b = PHASES["0"]  # Signal B phase
phase_assist = PHASES["0"]  # Assist signal phase

# Generate waves
wave_a = generate_wave(AMPLITUDE_A, phase_a, t)
wave_b = generate_wave(AMPLITUDE_B, phase_b, t)
assist_wave = generate_wave(AMPLITUDE_ASSIST, phase_assist, t)

# Combined wave
combined_wave = wave_a + wave_b + assist_wave

# Function to visualize selected signals
def visualize_signals(show_a=True, show_b=True, show_assist=True, show_combined=True):
    plt.figure(figsize=(12, 6))
    if show_a:
        plt.plot(t, wave_a, label="Signal A", alpha=0.7)
    if show_b:
        plt.plot(t, wave_b, label="Signal B", alpha=0.7)
    if show_assist:
        plt.plot(t, assist_wave, label="Assist Signal", alpha=0.7)
    if show_combined:
        plt.plot(t, combined_wave, label="Combined Wave", linewidth=2, color="black")
    plt.title("Interference Pattern for 0-0 Scenario", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.show()

# Example usage
visualize_signals(show_a=True, show_b=True, show_assist=True, show_combined=True)
