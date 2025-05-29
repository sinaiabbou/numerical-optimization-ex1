import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive mode
import matplotlib.pyplot as plt
import os
from src.unconstrained_min import UnconstrainedMinimizer
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
        self.max_iter_rosenbrock_gd = 10000  # Special case for Rosenbrock with GD
        self.x0_standard = np.array([1.1, 1.1])
        self.x0_rosenbrock = np.array([-1.0, 2.0])
        
        # Create output directory for plots
        self.output_dir = "test_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_comparison(self, func, newton_iterations, gd_iterations, title, filename_base):
        """Plot comparison between Newton and Gradient Descent methods"""
        # Plot 1: Contour with both paths
        plt.figure(figsize=(10, 8))
        
        # Define grid for contours
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = func(np.array([X[j, i], Y[j, i]]))[0]
        
        # Plot contours
        plt.contour(X, Y, Z, levels=20)
        
        # Plot paths
        newton_path = np.array(newton_iterations)
        gd_path = np.array(gd_iterations)
        plt.plot(newton_path[:, 0], newton_path[:, 1], 'r.-', label='Newton', linewidth=1)
        plt.plot(gd_path[:, 0], gd_path[:, 1], 'b.-', label='Gradient Descent', linewidth=1)
        
        plt.title(f"{title} - Optimization Paths")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{filename_base}_paths.png"))
        plt.close()
        
        # Plot 2: Function values vs iterations
        plt.figure(figsize=(10, 6))
        
        newton_values = [func(x)[0] for x in newton_iterations]
        gd_values = [func(x)[0] for x in gd_iterations]
        
        plt.semilogy(range(len(newton_values)), newton_values, 'r.-', label='Newton')
        plt.semilogy(range(len(gd_values)), gd_values, 'b.-', label='Gradient Descent')
        
        plt.title(f"{title} - Convergence")
        plt.xlabel('Iteration')
        plt.ylabel('Function Value (log scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{filename_base}_convergence.png"))
        plt.close()

    def test_quadratic1(self):
        """Test minimization of quadratic function with circular contours"""
        # Newton's method
        minimizer_newton = UnconstrainedMinimizer(method='newton')
        x_opt_newton, f_opt_newton, success_newton = minimizer_newton.minimize(
            quadratic1, self.x0_standard, self.obj_tol, self.param_tol, self.max_iter
        )
        
        # Gradient Descent
        minimizer_gd = UnconstrainedMinimizer(method='gradient_descent')
        x_opt_gd, f_opt_gd, success_gd = minimizer_gd.minimize(
            quadratic1, self.x0_standard, self.obj_tol, self.param_tol, self.max_iter
        )
        
        # Create plots
        self.plot_comparison(
            quadratic1,
            minimizer_newton.iterations,
            minimizer_gd.iterations,
            "Quadratic Function (Circular Contours)",
            "quadratic1"
        )
        
        # Verify results
        self.assertTrue(success_newton)
        self.assertTrue(success_gd)
        np.testing.assert_array_almost_equal(x_opt_newton, np.zeros(2), decimal=6)
        np.testing.assert_array_almost_equal(x_opt_gd, np.zeros(2), decimal=6)

    def test_quadratic2(self):
        """Test minimization of quadratic function with elliptical contours"""
        # Newton's method
        minimizer_newton = UnconstrainedMinimizer(method='newton')
        x_opt_newton, f_opt_newton, success_newton = minimizer_newton.minimize(
            quadratic2, self.x0_standard, self.obj_tol, self.param_tol, self.max_iter
        )
        
        # Gradient Descent
        minimizer_gd = UnconstrainedMinimizer(method='gradient_descent')
        x_opt_gd, f_opt_gd, success_gd = minimizer_gd.minimize(
            quadratic2, self.x0_standard, self.obj_tol, self.param_tol, self.max_iter
        )
        
        # Create plots
        self.plot_comparison(
            quadratic2,
            minimizer_newton.iterations,
            minimizer_gd.iterations,
            "Quadratic Function (Elliptical Contours)",
            "quadratic2"
        )
        
        # Verify results
        self.assertTrue(success_newton)
        self.assertTrue(success_gd)
        np.testing.assert_array_almost_equal(x_opt_newton, np.zeros(2), decimal=6)
        np.testing.assert_array_almost_equal(x_opt_gd, np.zeros(2), decimal=6)

    def test_quadratic3(self):
        """Test minimization of quadratic function with rotated elliptical contours"""
        # Newton's method
        minimizer_newton = UnconstrainedMinimizer(method='newton')
        x_opt_newton, f_opt_newton, success_newton = minimizer_newton.minimize(
            quadratic3, self.x0_standard, self.obj_tol, self.param_tol, self.max_iter
        )
        
        # Gradient Descent
        minimizer_gd = UnconstrainedMinimizer(method='gradient_descent')
        x_opt_gd, f_opt_gd, success_gd = minimizer_gd.minimize(
            quadratic3, self.x0_standard, self.obj_tol, self.param_tol, self.max_iter
        )
        
        # Create plots
        self.plot_comparison(
            quadratic3,
            minimizer_newton.iterations,
            minimizer_gd.iterations,
            "Quadratic Function (Rotated Elliptical Contours)",
            "quadratic3"
        )
        
        # Verify results
        self.assertTrue(success_newton)
        self.assertTrue(success_gd)
        np.testing.assert_array_almost_equal(x_opt_newton, np.zeros(2), decimal=6)
        np.testing.assert_array_almost_equal(x_opt_gd, np.zeros(2), decimal=6)

    def test_rosenbrock(self):
        """Test minimization of Rosenbrock function"""
        # Newton's method
        minimizer_newton = UnconstrainedMinimizer(method='newton')
        x_opt_newton, f_opt_newton, success_newton = minimizer_newton.minimize(
            rosenbrock_function, self.x0_rosenbrock, self.obj_tol, self.param_tol, self.max_iter
        )
        
        # Gradient Descent with 10000 iterations
        minimizer_gd = UnconstrainedMinimizer(method='gradient_descent')
        x_opt_gd, f_opt_gd, success_gd = minimizer_gd.minimize(
            rosenbrock_function, self.x0_rosenbrock, self.obj_tol, self.param_tol, self.max_iter_rosenbrock_gd
        )
        
        # Create plots
        self.plot_comparison(
            rosenbrock_function,
            minimizer_newton.iterations,
            minimizer_gd.iterations,
            "Rosenbrock Function",
            "rosenbrock"
        )
        
        # Verify results
        self.assertTrue(success_newton)
        np.testing.assert_array_almost_equal(x_opt_newton, np.ones(2), decimal=4)

    def test_exponential(self):
        """Test minimization of exponential function"""
        # Newton's method
        minimizer_newton = UnconstrainedMinimizer(method='newton')
        x_opt_newton, f_opt_newton, success_newton = minimizer_newton.minimize(
            exponential_function, self.x0_standard, self.obj_tol, self.param_tol, self.max_iter
        )
        
        # Gradient Descent
        minimizer_gd = UnconstrainedMinimizer(method='gradient_descent')
        x_opt_gd, f_opt_gd, success_gd = minimizer_gd.minimize(
            exponential_function, self.x0_standard, self.obj_tol, self.param_tol, self.max_iter
        )
        
        # Create plots
        self.plot_comparison(
            exponential_function,
            minimizer_newton.iterations,
            minimizer_gd.iterations,
            "Exponential Function",
            "exponential"
        )
        
        # Verify results
        self.assertTrue(success_newton)
        self.assertTrue(success_gd)
        np.testing.assert_array_almost_equal(x_opt_newton, np.zeros(2), decimal=6)
        np.testing.assert_array_almost_equal(x_opt_gd, np.zeros(2), decimal=6)

if __name__ == '__main__':
    unittest.main() 