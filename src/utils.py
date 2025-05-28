import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

def plot_contours_and_path(f: Callable,
                          iterations: List[np.ndarray],
                          title: str,
                          xlim: Tuple[float, float] = (-2, 2),
                          ylim: Tuple[float, float] = (-2, 2),
                          n_points: int = 100) -> None:
    """
    Plot contour lines of the objective function and optimization path.
    
    Args:
        f: Objective function
        iterations: List of points visited during optimization
        title: Plot title
        xlim: x-axis limits
        ylim: y-axis limits
        n_points: Number of points for contour plot grid
    """
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Evaluate function on grid
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j], _, _ = f(np.array([X[i, j], Y[i, j]]), False)
    
    # Create contour plot
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20)
    
    # Plot optimization path
    path = np.array(iterations)
    plt.plot(path[:, 0], path[:, 1], 'r.-', label='Optimization path')
    
    plt.title(title)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.colorbar(label='f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_convergence(f_values: List[float],
                    title: str) -> None:
    """
    Plot function values vs iteration number.
    
    Args:
        f_values: List of function values at each iteration
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(f_values)), f_values, 'b.-')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.yscale('log')
    plt.grid(True)
    plt.show() 