import numpy as np
from typing import Tuple, Optional

# ========= EXISTING FUNCTIONS (unchanged) =========

def quadratic_function(Q: np.ndarray) -> callable:
    def f(x: np.ndarray, compute_hessian: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        value = 0.5 * x.T @ Q @ x
        gradient = Q @ x
        hessian = Q if compute_hessian else None
        return value, gradient, hessian
    return f

def rosenbrock_function(x: np.ndarray, compute_hessian: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    x1, x2 = x[0], x[1]
    value = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    dx2 = 200 * (x2 - x1**2)
    gradient = np.array([dx1, dx2])
    if compute_hessian:
        h11 = -400 * (x2 - 3*x1**2) + 2
        h12 = -400 * x1
        h22 = 200
        hessian = np.array([[h11, h12], [h12, h22]])
    else:
        hessian = None
    return value, gradient, hessian

def linear_function(a: np.ndarray) -> callable:
    def f(x: np.ndarray, compute_hessian: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        value = a.T @ x
        gradient = a
        hessian = np.zeros((len(x), len(x))) if compute_hessian else None
        return value, gradient, hessian
    return f

def exponential_function(x: np.ndarray, compute_hessian: bool = False) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    x1, x2 = x[0], x[1]
    term1 = np.exp(x1 + 3*x2 - 0.1)
    term2 = np.exp(x1 - 3*x2 - 0.1)
    term3 = np.exp(-x1 - 0.1)
    value = term1 + term2 + term3
    dx1 = term1 + term2 - term3
    dx2 = 3*term1 - 3*term2
    gradient = np.array([dx1, dx2])
    if compute_hessian:
        h11 = term1 + term2 + term3
        h12 = 3*term1 - 3*term2
        h22 = 9*term1 + 9*term2
        hessian = np.array([[h11, h12], [h12, h22]])
    else:
        hessian = None
    return value, gradient, hessian

# ========= Q matrices (unchanged) =========

Q1 = np.array([[1, 0], [0, 1]])
Q2 = np.array([[1, 0], [0, 100]])
Q3 = np.array([[100, 0], [0, 1]])

quadratic1 = quadratic_function(Q1)
quadratic2 = quadratic_function(Q2)
quadratic3 = quadratic_function(Q3)

# ========= HW02 ADDITIONS =========

# --- Example QP ---

def qp_func_hw2(x: np.ndarray, compute_hessian: bool = True) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    # Objective: x^2 + y^2 + (z+1)^2
    val = x[0]**2 + x[1]**2 + (x[2]+1)**2
    grad = np.array([2*x[0], 2*x[1], 2*(x[2]+1)])
    hess = np.diag([2,2,2]) if compute_hessian else None
    return val, grad, hess

def qp_constraints_hw2():
    # Constraints: x >=0, y >=0, z >=0
    def g1(x):
        val = x[0]
        grad = np.array([1,0,0])
        hess = np.zeros((3,3))
        return val, grad, hess

    def g2(x):
        val = x[1]
        grad = np.array([0,1,0])
        hess = np.zeros((3,3))
        return val, grad, hess

    def g3(x):
        val = x[2]
        grad = np.array([0,0,1])
        hess = np.zeros((3,3))
        return val, grad, hess

    return [g1, g2, g3]

qp_eq_mat_hw2 = np.array([[1,1,1]])
qp_eq_rhs_hw2 = np.array([1])
qp_x0_hw2 = np.array([0.1,0.2,0.7])

# --- Example LP ---

def lp_func_hw2(x: np.ndarray, compute_hessian: bool = True) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    # Objective: maximize x + y <=> minimize -x - y
    val = -x[0] - x[1]
    grad = np.array([-1,-1])
    hess = np.zeros((2,2)) if compute_hessian else None
    return val, grad, hess

def lp_constraints_hw2():
    def g1(x):
        val = x[0] + x[1] -1
        grad = np.array([1,1])
        hess = np.zeros((2,2))
        return val, grad, hess

    def g2(x):
        val = 1 - x[1]
        grad = np.array([0,-1])
        hess = np.zeros((2,2))
        return val, grad, hess

    def g3(x):
        val = 2 - x[0]
        grad = np.array([-1,0])
        hess = np.zeros((2,2))
        return val, grad, hess

    def g4(x):
        val = x[1]
        grad = np.array([0,1])
        hess = np.zeros((2,2))
        return val, grad, hess

    return [g1,g2,g3,g4]

lp_x0_hw2 = np.array([0.5,0.75])