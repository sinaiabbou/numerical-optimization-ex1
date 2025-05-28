import numpy as np
from typing import Tuple, Optional

def quadratic_function(Q: np.ndarray) -> callable:
    """
    Creates a quadratic function f(x) = x^T Q x
    
    Args:
        Q: Symmetric positive definite matrix
    
    Returns:
        Function that computes value, gradient and Hessian
    """
    def f(x: np.ndarray, compute_hessian: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        value = 0.5 * x.T @ Q @ x
        gradient = Q @ x
        hessian = Q if compute_hessian else None
        return value, gradient, hessian
    return f

def rosenbrock_function(x: np.ndarray, compute_hessian: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    """
    Rosenbrock function: f(x) = 100(x₂ - x₁²)² + (1 - x₁)²
    """
    x1, x2 = x[0], x[1]
    
    # Function value
    value = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    
    # Gradient
    dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    dx2 = 200 * (x2 - x1**2)
    gradient = np.array([dx1, dx2])
    
    # Hessian (if requested)
    if compute_hessian:
        h11 = -400 * (x2 - 3*x1**2) + 2
        h12 = -400 * x1
        h22 = 200
        hessian = np.array([[h11, h12], [h12, h22]])
    else:
        hessian = None
    
    return value, gradient, hessian

def linear_function(a: np.ndarray) -> callable:
    """
    Creates a linear function f(x) = a^T x
    
    Args:
        a: Direction vector
    
    Returns:
        Function that computes value, gradient and Hessian
    """
    def f(x: np.ndarray, compute_hessian: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        value = a.T @ x
        gradient = a
        hessian = np.zeros((len(x), len(x))) if compute_hessian else None
        return value, gradient, hessian
    return f

def exponential_function(x: np.ndarray, compute_hessian: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    """
    Function f(x₁,x₂) = e^(x₁+3x₂-0.1) + e^(x₁-3x₂-0.1) + e^(-x₁-0.1)
    """
    x1, x2 = x[0], x[1]
    
    # Compute terms
    term1 = np.exp(x1 + 3*x2 - 0.1)
    term2 = np.exp(x1 - 3*x2 - 0.1)
    term3 = np.exp(-x1 - 0.1)
    
    # Function value
    value = term1 + term2 + term3
    
    # Gradient
    dx1 = term1 + term2 - term3
    dx2 = 3*term1 - 3*term2
    gradient = np.array([dx1, dx2])
    
    # Hessian (if requested)
    if compute_hessian:
        h11 = term1 + term2 + term3
        h12 = 3*term1 - 3*term2
        h22 = 9*term1 + 9*term2
        hessian = np.array([[h11, h12], [h12, h22]])
    else:
        hessian = None
    
    return value, gradient, hessian

# Example Q matrices as specified in the assignment
Q1 = np.array([[1, 0], [0, 1]])  # Circles
Q2 = np.array([[1, 0], [0, 100]])  # Axis-aligned ellipses
Q3 = np.array([[100, 0], [0, 1]])  # Rotated ellipses

# Create the quadratic functions
quadratic1 = quadratic_function(Q1)
quadratic2 = quadratic_function(Q2)
quadratic3 = quadratic_function(Q3) 