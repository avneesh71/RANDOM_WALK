import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
n_steps = 1000     # Number of steps in each random walk
n_walks = 100000  # Number of random walks

# Initialize arrays to store positions and mean square displacement
final_positions = np.zeros((n_walks, 3))
all_positions = np.zeros((n_walks, n_steps + 1, 3))
mean_displacement = np.zeros((n_steps + 1, 3))
mean_square_displacement = np.zeros((n_steps + 1, 3))

# Simulate random walks
for i in range(n_walks):
    steps = np.random.choice([-1, 1], size=(n_steps, 3))
    positions = np.cumsum(steps, axis=0)
    all_positions[i, 1:, :] = positions  # Store all positions for each walk
    final_positions[i, :] = positions[-1, :]  # Store the final position

# Calculate mean displacement and mean square displacement
for t in range(n_steps + 1):
    mean_displacement[t, :] = np.mean(all_positions[:, t, :], axis=0)
    mean_square_displacement[t, :] = np.mean(all_positions[:, t, :] ** 2, axis=0)

# Plot Mean Displacement vs. Step Number
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(mean_displacement[:, 0], label='X direction')
plt.plot(mean_displacement[:, 1], label='Y direction')
plt.plot(mean_displacement[:, 2], label='Z direction')
plt.title('Mean Displacement vs. Step Number')
plt.xlabel('Step Number')
plt.ylabel('Mean Displacement')
plt.legend()

# Plot Mean Square Displacement vs. Step Number
plt.subplot(1, 3, 2)
plt.plot(mean_square_displacement[:, 0], label='X direction')
plt.plot(mean_square_displacement[:, 1], label='Y direction')
plt.plot(mean_square_displacement[:, 2], label='Z direction')
plt.title('Mean Square Displacement vs. Step Number')
plt.xlabel('Step Number')
plt.ylabel('Mean Square Displacement')
plt.legend()

# Plot Probability Distribution of Final Positions in 3D
plt.subplot(1, 3, 3, projection='3d')
ax = plt.gca()
scat = ax.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], c='blue', alpha=0.1, s=1)
ax.set_title('Probability Distribution of Final Positions')
ax.set_xlabel('Final Position X')
ax.set_ylabel('Final Position Y')
ax.set_zlabel('Final Position Z')

plt.tight_layout()
plt.show()
