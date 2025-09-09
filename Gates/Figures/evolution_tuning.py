import matplotlib.pyplot as plt
import numpy as np

def plot_gate_bar(
    inputs,
    outputs,
    ylabel="Normalized Output",
    xlabel="Input",
    threshold: float = 0.5,
    label_height: float = 0.6,
    savepath: str = None   # new optional argument
):
    """
    Create a bar plot for logic gate inputs/outputs.
    - inputs: list of (A, B) tuples
    - outputs: list of output values (floats)
    - savepath: filename to save the figure (if provided)
    """

    outputs = np.array(outputs, dtype=float)
    if outputs.max() > 1.0:
        outputs = outputs / outputs.max()

    fig, ax = plt.subplots(figsize=(5,4))
    bars = ax.bar(range(len(inputs)), outputs, color="skyblue")

    # Add horizontal threshold line
    ax.axhline(y=threshold, color="red", linestyle="--")
    ax.set_ylim(0, 1.1)

    ax.set_ylabel(ylabel, fontsize=17)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xticks([])

    # Add compact input labels like (0,0)
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_height,
            f"({inputs[i][0]},{inputs[i][1]})",
            ha="center", va="bottom", fontsize=16, color="black"
        )

    plt.tight_layout()

    # Save if a path is provided
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)  # close to free memory if many plots
    else:
        plt.show()


if __name__ == "__main__":
    inputs = [(0,0), (0,1), (1,0), (1,1)]

    all_outputs = [
        [0.5, 0.45, 0.8, 0.34],
        [0.35, 0.4, 0.7, 0.4],
        [0.3, 0.4, 0.5, 0.6],
        [0.3, 0.32, 0.45, 0.7],
        [0.2, 0.3, 0.3, 0.75],
        [0.1, 0.14, 0.15, 1]
    ]

    # Generate and save each plot
    for i, outputs in enumerate(all_outputs, start=1):
        filename = f"plot_{i}.png"
        plot_gate_bar(inputs, outputs, savepath=filename)
        print(f"Saved {filename}")
