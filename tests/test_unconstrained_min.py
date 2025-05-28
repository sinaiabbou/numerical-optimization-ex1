import unittest
import numpy as np
from src.unconstrained_min import UnconstrainedMinimizer
from src.utils import plot_contours_and_path, plot_convergence
from tests.examples import (
    quadratic1, quadratic2, quadratic3,
    rosenbrock_function,
    linear_function,
    exponential_function
)

class TestUnconstrainedMinimizer(unittest.TestCase):
    def setUp(self):
        """Set up test parameters"""
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 100
        self.x0_standard = np.array([1.1, 1.1])
        self.x0_rosenbrock = np.array([-1.2, 1.0])
        
    def test_quadratic1(self):
        """Test minimization of quadratic function with circular contours"""
        minimizer = UnconstrainedMinimizer(method='newton')
        x_opt, f_opt, success = minimizer.minimize(
            quadratic1,
            self.x0_standard,
            self.obj_tol,
            self.param_tol,
            self.max_iter
        )
        
        # Plot results
        plot_contours_and_path(quadratic1, minimizer.iterations,
                             "Quadratic Function (Circular Contours) - Newton Method")
        plot_convergence(minimizer.f_values,
                       "Convergence - Quadratic Function (Circular) - Newton Method")
        
        # Check results
        self.assertTrue(success)
        np.testing.assert_array_almost_equal(x_opt, np.zeros(2), decimal=6)
        self.assertAlmostEqual(f_opt, 0.0, places=10)
        
    def test_rosenbrock(self):
        """Test minimization of Rosenbrock function"""
        # Test with Newton method
        minimizer_newton = UnconstrainedMinimizer(method='newton')
        x_opt_newton, f_opt_newton, success_newton = minimizer_newton.minimize(
            rosenbrock_function,
            self.x0_rosenbrock,
            self.obj_tol,
            self.param_tol,
            self.max_iter
        )
        
        # Test with Gradient Descent
        minimizer_gd = UnconstrainedMinimizer(method='gradient_descent')
        x_opt_gd, f_opt_gd, success_gd = minimizer_gd.minimize(
            rosenbrock_function,
            self.x0_rosenbrock,
            self.obj_tol,
            self.param_tol,
            10000  # More iterations for GD on Rosenbrock
        )
        
        # Plot results
        plot_contours_and_path(rosenbrock_function, minimizer_newton.iterations,
                             "Rosenbrock Function - Newton Method")
        plot_contours_and_path(rosenbrock_function, minimizer_gd.iterations,
                             "Rosenbrock Function - Gradient Descent")
        
        # Check results
        self.assertTrue(success_newton)
        np.testing.assert_array_almost_equal(x_opt_newton, np.ones(2), decimal=4)
        self.assertLess(f_opt_newton, 1e-8)

if __name__ == '__main__':
    unittest.main() 