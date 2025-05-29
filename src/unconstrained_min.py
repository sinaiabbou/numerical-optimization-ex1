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
        self.iterations: List[np.ndarray] = []
        self.f_values: List[float] = []
    
    def minimize(self, 
                f: Callable,
                x0: np.ndarray,
                obj_tol: float = 1e-12,
                param_tol: float = 1e-8,
                max_iter: int = 100) -> Tuple[np.ndarray, float, bool]:
        """
        Minimize an unconstrained objective function using line search.
        
        Args:
            f: The objective function that returns (f, g, h) - value, gradient, and Hessian
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
        f_val, g, h = f(x, True)
        self.iterations = [x.copy()]
        self.f_values = [f_val]
        
        for i in range(max_iter):
            if self.method == 'newton' and h is not None:
                d = np.linalg.solve(h, -g)
            else:
                d = -g
            alpha = self._backtracking_line_search(f, x, f_val, g, d)
            x_new = x + alpha * d
            f_new, g_new, h_new = f(x_new, True)
            self.iterations.append(x_new.copy())
            self.f_values.append(f_new)
            print(f"Iteration {i}: x = {x_new}, f(x) = {f_new}")
            if abs(f_new - f_val) < obj_tol:
                print(f"Converged: obj_tol reached (|f_new - f_old| = {abs(f_new - f_val):.2e})")
                return x_new, f_new, True
            if np.linalg.norm(x_new - x) < param_tol:
                print(f"Converged: param_tol reached (|x_new - x| = {np.linalg.norm(x_new - x):.2e})")
                return x_new, f_new, True
            x, f_val, g, h = x_new, f_new, g_new, h_new
        print("Max iterations reached without convergence.")
        return x, f_val, False

    def _backtracking_line_search(self,
                                f: Callable,
                                x: np.ndarray,
                                f_x: float,
                                g: np.ndarray,
                                d: np.ndarray,
                                alpha: float = 1.0,
                                rho: float = 0.5,
                                c: float = 1e-4) -> float:
        """
        Implements backtracking line search with Armijo condition only.
        """
        g_d = g.dot(d)
        while True:
            x_new = x + alpha * d
            f_new, _, _ = f(x_new, False)
            if f_new <= f_x + c * alpha * g_d:
                return alpha
            alpha *= rho 