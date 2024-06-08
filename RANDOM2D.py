import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Parameters
n_steps = 1000     # Number of steps in each random walk
n_walks = 10000  # Number of random walks

# Initialize arrays to store positions and mean square displacement
final_positions = np.zeros((n_walks, 2))
all_positions = np.zeros((n_walks, n_steps + 1, 2))
mean_displacement = np.zeros((n_steps + 1, 2))
mean_square_displacement = np.zeros((n_steps + 1, 2))

# Simulate random walks
for i in range(n_walks):
    steps = np.random.choice([-1, 1], size=(n_steps, 2))
    positions = np.cumsum(steps, axis=0)
    all_positions[i, 1:, :] = positions  # Store all positions for each walk
    final_positions[i, :] = positions[-1, :]  # Store the final position

# Calculate mean displacement and mean square displacement
for t in range(n_steps + 1):
    mean_displacement[t, :] = np.mean(all_positions[:, t, :], axis=0)
    mean_square_displacement[t, :] = np.mean(all_positions[:, t, :] ** 2, axis=0)

# Probability distribution of final positions
unique_positions, counts = np.unique(final_positions, axis=0, return_counts=True)
probability_distribution = counts / n_walks

# Plot Mean Displacement vs. Step Number
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(mean_displacement[:, 0], label='X direction')
plt.plot(mean_displacement[:, 1], label='Y direction')
plt.title('Mean Displacement vs. Step Number')
plt.xlabel('Step Number')
plt.ylabel('Mean Displacement')
plt.legend()

# Plot Mean Square Displacement vs. Step Number
plt.subplot(1, 3, 2)
plt.plot(mean_square_displacement[:, 0], label='X direction')
plt.plot(mean_square_displacement[:, 1], label='Y direction')
plt.title('Mean Square Displacement vs. Step Number')
plt.xlabel('Step Number')
plt.ylabel('Mean Square Displacement')
plt.legend()

# Plot Probability Distribution of Final Positions
plt.subplot(1, 3, 3)
plt.scatter(unique_positions[:, 0], unique_positions[:, 1], c=probability_distribution, cmap='viridis', s=1)
plt.title('Probability Distribution of Final Positions')
plt.xlabel('Final Position X')
plt.ylabel('Final Position Y')
plt.colorbar(label='Probability')

plt.tight_layout()
plt.show()

# Save the plot to a file
plt.savefig('random_walk_plot.png')

# Open the saved image using the default image viewer
subprocess.run(['xdg-open', 'random_walk_plot.png'])
