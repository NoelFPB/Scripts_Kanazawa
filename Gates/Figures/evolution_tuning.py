import matplotlib.pyplot as plt
import numpy as np

def plot_gate_bar(
    inputs,            # list of tuples (A, B)
    outputs,           # list of float values (same length as inputs)
    title="Logic Gate",
    ylabel="Normalized Output",
    xlabel="Input",
    threshold: float = 0.5
):
    """
    Create a bar plot for logic gate inputs/outputs.
    - inputs: list of (A, B) tuples
    - outputs: list of output values (floats)
    """

    labels = [f"A={a}\nB={b}" for a, b in inputs]
    outputs = np.array(outputs, dtype=float)
    if outputs.max() > 1.0:
        outputs = outputs / outputs.max()

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(labels, outputs, color="skyblue")
    ax.axhline(y=threshold, color="red", linestyle="--")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define the input combinations
    inputs = [(0,0), (0,1), (1,0), (1,1)]

    # Now define multiple sets of outputs (each list = one plot)
    all_outputs = [
        [0.5, 0.45, 0.8, 0.34],   # First iteration
        [0.35, 0.4, 0.7, 0.4],  # Second iteration
        [0.3, 0.4, 0.5, 0.6],   # Third iteration
        [0.3, 0.32, 0.45, 0.7],
        [0.2, 0.3, 0.3, 0.75],
        [0.1, 0.14, 0.15, 0.95]
    ]

    # Generate a plot for each set
    for i, outputs in enumerate(all_outputs, start=1):
        plot_gate_bar(
            inputs,
            outputs,
            title=f"AND Gate"
        )
