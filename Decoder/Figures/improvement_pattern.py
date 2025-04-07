import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Simulated example of total score improvement up to a value close to 60
iterations = list(range(1, 21))
example_scores = [
    12.55, 15.57, 16.91, 19.79, 20.0,
    28.22, 31.53, 35.48, 38.12, 42.1,
    46.0, 49.0, 52.0, 54.5, 55.5,
    56.0, 57.0, 58.0, 59.0, 60.0
]


# example_scores = [
#      12.55, 15.57, 16.91, 19.79, 20.0]

plt.figure(figsize=(10, 7))
plt.plot(iterations, example_scores, marker='o')
#plt.title("Example of Total Score Growing During Optimization")
plt.xlabel("Iteration", fontsize=18)
plt.ylabel("Total Score", fontsize=18)
plt.grid(True)
plt.xticks(iterations, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 65)
#plt.tight_layout()
plt.savefig("total_score_plot.svg", format="svg")
plt.show()
