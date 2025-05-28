import numpy as np
from typing import Callable, Tuple, List, Optional

class UnconstrainedMinimizer:
    def __init__(self, method: str = 'newton'):
        """
        Initialize the unconstrained minimizer.
        
        Args:
            method (str): The optimization method to use ('newton' or 'gradient_descent')
        """
        self.method = method.lower()
        self.iterations = []
        self.f_values = []
        
    def minimize(self, 
                f: Callable,
                x0: np.ndarray,
                obj_tol: float = 1e-12,
                param_tol: float = 1e-8,
                max_iter: int = 100) -> Tuple[np.ndarray, float, bool]:
        """
        Minimize an unconstrained objective function using line search.
        
        Args:
            f: The objective function that returns (f, g, h) - value, gradient, and optionally Hessian
            x0: Initial point
            obj_tol: Tolerance for change in objective value
            param_tol: Tolerance for parameter changes
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple containing:
            - Final solution point
            - Final objective value
            - Success flag
        """
        x = x0.copy()
        f_val, g, h = f(x, True)  # Initial evaluation
        self.iterations = [x.copy()]
        self.f_values = [f_val]
        
        for i in range(max_iter):
            # Get search direction based on method
            if self.method == 'newton' and h is not None:
                # Solve system instead of inverting matrix
                d = np.linalg.solve(h, -g)
            else:  # gradient descent
                d = -g
                
            # Backtracking line search with Wolfe conditions
            alpha = self._backtracking_line_search(f, x, f_val, g, d)
            
            # Update position
            x_new = x + alpha * d
            f_new, g_new, h_new = f(x_new, True)
            
            # Store iteration history
            self.iterations.append(x_new.copy())
            self.f_values.append(f_new)
            
            # Print iteration info
            print(f"Iteration {i}: x = {x_new}, f(x) = {f_new}")
            
            # Check convergence
            if abs(f_new - f_val) < obj_tol:
                return x_new, f_new, True
            if np.linalg.norm(x_new - x) < param_tol:
                return x_new, f_new, True
                
            # Update for next iteration
            x, f_val, g, h = x_new, f_new, g_new, h_new
            
        return x, f_val, False
    
    def _backtracking_line_search(self,
                                f: Callable,
                                x: np.ndarray,
                                f_x: float,
                                g: np.ndarray,
                                d: np.ndarray,
                                alpha: float = 1.0,
                                rho: float = 0.5,
                                c: float = 0.01) -> float:
        """
        Implements backtracking line search with Wolfe conditions.
        
        Args:
            f: Objective function
            x: Current point
            f_x: Current function value
            g: Current gradient
            d: Search direction
            alpha: Initial step size
            rho: Backtracking factor
            c: Wolfe condition constant
            
        Returns:
            Step size that satisfies Wolfe conditions
        """
        g_d = g.dot(d)  # Directional derivative
        
        while True:
            x_new = x + alpha * d
            f_new, _, _ = f(x_new, False)
            
            # Armijo condition
            if f_new <= f_x + c * alpha * g_d:
                return alpha
                
            alpha *= rho 