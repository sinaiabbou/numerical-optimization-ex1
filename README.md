# HW02 – Interior Point Method

Author: Sinai Abbou

## Project Structure

ex1_programming_nop/
    src/
        __pycache__/
        __init__.py
        constrained_min.py
        unconstrained_min.py
        utils.py
    test_results/
    tests/
        __pycache__/
        __init__.py
        examples.py
        plot_hw02.py
        test_constrained_min.py
        test_unconstrained_min.py
    venv/
    .gitignore
    lp_feasible_and_path.png
    lp_objective_vs_iteration.png
    qp_feasible_and_path.png
    qp_objective_vs_iteration.png
    README.md
    requirements.txt

## Usage

- Run tests:
    python -m unittest tests/test_constrained_min.py

- Generate plots/results:
    python tests/plot_hw02.py

All results and plots are saved in the project root.