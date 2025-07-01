# HW02 – Interior Point Method

Author: Sinai Abbou

## Important files (HW02)

- `src/constrained_min.py` – Interior point (log-barrier) method implementation for constrained optimization.
- `src/unconstrained_min.py` – Unconstrained minimization (from HW01, required for the solver).
- `tests/examples.py` – Defines LP and QP problems for HW02.
- `tests/test_constrained_min.py` – Unit tests for HW02 (LP and QP).
- `tests/plot_hw02.py` – Script to generate required plots and print solution/constraint info.
- `lp_feasible_and_path.png`, `lp_objective_vs_iteration.png`, `qp_feasible_and_path.png`, `qp_objective_vs_iteration.png` – Plots generated for HW02 report.

## How to use

- Run all tests:  
  `python -m unittest tests/test_constrained_min.py`

- Generate plots and print final results:  
  `python tests/plot_hw02.py`

All results and plots are saved in the project root.

**GitHub repo:** https://github.com/sinaiabbou/numerical-optimization-ex1