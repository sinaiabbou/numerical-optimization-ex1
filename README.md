# Numerical Optimization - Programming Assignment 01

This repository contains the implementation of an unconstrained optimization solver supporting both Newton and Gradient Descent methods with backtracking line search.

## Project Structure

```
.
├── src/
│   ├── unconstrained_min.py  # Main solver implementation
│   └── utils.py              # Plotting and utility functions
└── tests/
    ├── examples.py           # Test functions (quadratic, Rosenbrock, etc.)
    └── test_unconstrained_min.py  # Unit tests
```

## Features

- Modular implementation of unconstrained optimization solver
- Supports both Newton and Gradient Descent methods
- Backtracking line search with Wolfe conditions
- Visualization of optimization paths and convergence
- Test suite with various objective functions:
  - Quadratic functions (circular, axis-aligned ellipses, rotated ellipses)
  - Rosenbrock function
  - Linear function
  - Exponential function

## Dependencies

- NumPy
- Matplotlib

## Usage

To run the tests:

```bash
python -m unittest tests/test_unconstrained_min.py
```

## Implementation Details

- The solver never inverts matrices directly, instead using linear system solvers
- Generic implementation supporting arbitrary dimensions (not limited to 2D)
- Modular design allowing easy addition of new search direction methods 