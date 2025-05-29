from src.unconstrained_min import UnconstrainedMinimizer
from tests.examples import linear_function
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive mode
import matplotlib.pyplot as plt

# Starting point for linear function (as per instructions)
x0 = np.array([1.0, 1.0])

# Instantiate the linear function with a = [1, 1]
linfun = linear_function(np.array([1.0, 1.0]))

# Create the minimizer with gradient descent method
minimizer = UnconstrainedMinimizer(method='gradient_descent')

# Minimize the linear function with 100 max iterations (default for GD)
x_opt, f_opt, success = minimizer.minimize(
    f=linfun,
    x0=x0,
    obj_tol=1e-12,
    param_tol=1e-8,
    max_iter=100
)

print("Results of optimization with Gradient Descent on linear function (100 iterations max):")
print(f"Initial point: {x0}")
print(f"Optimal point found: {x_opt}")
print(f"Minimum value found: {f_opt}")
print(f"Success?: {success}")
print(f"\nNumber of iterations: {len(minimizer.iterations)}")

from src.utils import plot_contours_and_path

plt.figure(figsize=(8, 6))

plot_contours_and_path(
    f=linfun,
    iterations=minimizer.iterations,
    title="Minimization of linear function (Gradient Descent, 100 iter.)",
    xlim=(-2, 2),
    ylim=(-2, 2)
)

import os
output_file = os.path.join(os.getcwd(), 'linear_optimization_gd.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close('all')  # Close all figures

print(f"\nThe plot has been saved to:\n{output_file}") 