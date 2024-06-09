import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import subprocess

def simulate_random_walk(dim, n_steps, n_walks):
    final_positions = np.zeros((n_walks, dim))
    all_positions = np.zeros((n_walks, n_steps + 1, dim))
    mean_displacement = np.zeros((n_steps + 1, dim))
    mean_square_displacement = np.zeros((n_steps + 1, dim))

    for i in range(n_walks):
        steps = np.random.choice([-1, 1], size=(n_steps, dim)) # Random steps
        positions = np.cumsum(steps, axis=0) # Cumulative sum to get positions
        all_positions[i, 1:] = positions
        final_positions[i] = positions[-1]

    for t in range(n_steps + 1):
        for d in range(dim):
            mean_displacement[t, d] = np.mean(all_positions[:, t, d])
            mean_square_displacement[t, d] = np.mean(all_positions[:, t, d] ** 2)

    return final_positions, all_positions, mean_displacement, mean_square_displacement


def plot_results(dim, mean_displacement, mean_square_displacement, final_positions, n_walks):
    plt.figure(figsize=(18, 6))

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        if i == 0:
            plt.plot(mean_displacement)
            plt.title('Mean Displacement vs. Step Number')
            plt.ylabel('Mean Displacement')
        elif i == 1:
            plt.plot(mean_square_displacement)
            plt.title('Mean Square Displacement vs. Step Number')
            plt.ylabel('Mean Square Displacement')
        else:
            if dim == 1:
                plt.hist(final_positions, bins='auto', density=True, rwidth=0.8)
                plt.title('Probability Distribution of Final Positions')
                plt.xlabel('Final Position')
                plt.ylabel('Probability')
            elif dim == 2:
                plt.scatter(final_positions[:, 0], final_positions[:, 1], c='blue', alpha=0.1, s=1)
                plt.title('Probability Distribution of Final Positions')
                plt.xlabel('Final Position X')
                plt.ylabel('Final Position Y')
            elif dim == 3:
                ax = plt.subplot(1, 3, 3, projection='3d')  # Create a new subplot with 3D projection
                ax.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], c='blue', alpha=0.1, s=1)
                ax.set_title('Probability Distribution of Final Positions')
                ax.set_xlabel('Final Position X')
                ax.set_ylabel('Final Position Y')
                ax.set_zlabel('Final Position Z')

    plt.tight_layout()
    plt.show()

def main():
    while True:
        try:
            # Parameters
            dim = int(input("Enter dimension (1 for 1D, 2 for 2D, 3 for 3D): "))
            if dim < 1 or dim > 3:
                raise ValueError("Dimension must be 1, 2, or 3.")

            n_steps = 1000     # Number of steps in each random walk
            n_walks = 10000    # Number of random walks

            final_positions, all_positions, mean_displacement, mean_square_displacement = simulate_random_walk(dim, n_steps, n_walks)
            plot_results(dim, mean_displacement, mean_square_displacement, final_positions, n_walks)

            # Save the plot to a file
            #plt.savefig('random_walk_plot.png')

            # Open the saved image using the default image viewer
            #subprocess.run(['xdg-open', 'random_walk_plot.png'])

        except ValueError as ve:
            print("Error:", ve)
        except Exception as e:
            print("An unexpected error occurred:", e)

        while True:
            repeat = input("Do you want to run the simulation again? (y/n): ")
            if repeat.lower() == 'n':
                print("Bye!!")
                break
            elif repeat.lower() == 'y':
                break
            else:
                print("Please enter either 'y' or 'n'.\n")

        if repeat.lower() != 'y':
            break

if __name__ == "__main__":
    main()
