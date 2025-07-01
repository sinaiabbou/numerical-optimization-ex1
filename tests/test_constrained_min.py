import unittest
import numpy as np
from src.constrained_min import interior_pt
from tests.examples import (
    qp_func_hw2, qp_constraints_hw2, qp_eq_mat_hw2, qp_eq_rhs_hw2, qp_x0_hw2,
    lp_func_hw2, lp_constraints_hw2, lp_x0_hw2
)

class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        # HW02 QP example 
        x_opt, f_opt, success, hist = interior_pt(
            qp_func_hw2,
            qp_constraints_hw2(),
            qp_eq_mat_hw2,
            qp_eq_rhs_hw2,
            qp_x0_hw2
        )
        print("HW02 QP solution x:", x_opt)
        print("HW02 QP objective value:", f_opt)
        print("HW02 QP success:", success)
        self.assertTrue(success)

    def test_lp(self):
        # HW02 LP example 
        x_opt, f_opt, success, hist = interior_pt(
            lp_func_hw2,
            lp_constraints_hw2(),
            np.array([]),
            np.array([]),
            lp_x0_hw2
        )
        print("HW02 LP solution x:", x_opt)
        print("HW02 LP objective value (max):", -f_opt)
        print("HW02 LP success:", success)
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()