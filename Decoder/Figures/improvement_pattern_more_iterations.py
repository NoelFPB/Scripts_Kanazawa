import numpy as np
import matplotlib.pyplot as plt

iterations = list(range(1, 21))
example_scores = [
    12.55, 15.57, 16.91, 19.79, 20.0,
    28.22, 31.53, 35.48, 38.12, 42.1,
    46.0, 49.0, 52.0, 54.5, 55.5,
    56.0, 57.0, 58.0, 59.0, 60.0
]

plt.figure(figsize=(10, 7))
plt.plot(iterations, example_scores, marker='o', linewidth=2)

plt.xlabel("Iteration", fontsize=18)
plt.ylabel("Total Score", fontsize=18)
plt.grid(True)

# Show ticks at every iteration
all_ticks = list(range(1, 21))
# Show labels only every 5
tick_labels = [str(i) if i % 5 == 0 else "" for i in all_ticks]
plt.xticks(ticks=all_ticks, labels=tick_labels, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 65)

# Save as editable SVG
plt.savefig("total_score_plot.svg", format="svg")

plt.show()
