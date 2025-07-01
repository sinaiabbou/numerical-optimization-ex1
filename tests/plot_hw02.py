import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.constrained_min import interior_pt
from tests.examples import (
    qp_func_hw2, qp_constraints_hw2, qp_eq_mat_hw2, qp_eq_rhs_hw2, qp_x0_hw2,
    lp_func_hw2, lp_constraints_hw2, lp_x0_hw2
)

# ------------------------
# LP EXAMPLE: 2D POLYGON
# ------------------------

ineq_constraints_lp = lp_constraints_hw2()
eq_constraints_mat_lp = np.zeros((0, 2))
eq_constraints_rhs_lp = np.zeros(0)
x0_lp = lp_x0_hw2

x_hist_lp = []
obj_hist_lp = []

def collect_lp(x, fval):
    x_hist_lp.append(x.copy())
    obj_hist_lp.append(-fval)  # Since we minimize -x-y

x_lp, f_lp, success_lp, history_lp = interior_pt(
    lp_func_hw2, ineq_constraints_lp, eq_constraints_mat_lp, eq_constraints_rhs_lp, x0_lp
)
for x, f in history_lp:
    collect_lp(x, f)

print(f"LP Final objective (max): {-f_lp}")
print(f"LP Success: {success_lp}")

# Plot 1: Feasible region and central path
fig, ax = plt.subplots()
# Polygon: (intersection of constraints)
polygon = np.array([
    [0, 1],    # y = 1, x = 0
    [0.5, 0.5],# y = -x+1, x=0.5
    [2, -1],   # y = -x+1, x=2
    [2, 1]     # y=1, x=2
])
ax.plot([0, 0, 2, 2, 0], [0, 1, 1, 0, 0], 'k', label='Feasible region')
ax.plot(*zip(*[p[:2] for p in x_hist_lp]), 'ro-', label='Central path')
ax.plot(x_lp[0], x_lp[1], 'bs', label='Final solution')
ax.set_xlim(0, 2.1)
ax.set_ylim(0, 1.1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('LP feasible region and central path')
ax.legend()
plt.tight_layout()
plt.savefig('lp_feasible_and_path.png')

# Plot 2: Objective vs iteration
plt.figure()
plt.plot(obj_hist_lp, 'bo-')
plt.title("LP Objective vs Outer Iterations")
plt.xlabel("Outer iteration")
plt.ylabel("Objective value (max)")
plt.tight_layout()
plt.savefig('lp_objective_vs_iteration.png')

# ------------------------
# QP EXAMPLE: SIMPLEX IN 3D
# ------------------------

ineq_constraints_qp = qp_constraints_hw2()
eq_constraints_mat_qp = qp_eq_mat_hw2
eq_constraints_rhs_qp = qp_eq_rhs_hw2
x0_qp = qp_x0_hw2

x_hist_qp = []
obj_hist_qp = []

def collect_qp(x, fval):
    x_hist_qp.append(x.copy())
    obj_hist_qp.append(fval)

x_qp, f_qp, success_qp, history_qp = interior_pt(
    qp_func_hw2, ineq_constraints_qp, eq_constraints_mat_qp, eq_constraints_rhs_qp, x0_qp
)
for x, f in history_qp:
    collect_qp(x, f)

print(f"QP Final objective: {f_qp}")
print(f"QP Success: {success_qp}")

# Plot 3: Feasible region (simplex) and central path in 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the simplex face x+y+z=1, x>=0,y>=0,z>=0
verts = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])
ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], color='cyan', alpha=0.2)

hist_arr = np.array(x_hist_qp)
ax.plot(hist_arr[:,0], hist_arr[:,1], hist_arr[:,2], 'ro-', label='Central path')
ax.plot([x_qp[0]], [x_qp[1]], [x_qp[2]], 'bs', label='Final solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('QP feasible region and central path')
ax.legend()
plt.tight_layout()
plt.savefig('qp_feasible_and_path.png')

# Plot 4: Objective vs iteration (QP)
plt.figure()
plt.plot(obj_hist_qp, 'bo-')
plt.title("QP Objective vs Outer Iterations")
plt.xlabel("Outer iteration")
plt.ylabel("Objective value")
plt.tight_layout()
plt.savefig('qp_objective_vs_iteration.png')

# ------------------------
# PRINT CONSTRAINTS CHECK
# ------------------------

print("\n==== LP Final constraints check ====")
print("x + y =", x_lp[0] + x_lp[1], ">= 1 :", x_lp[0] + x_lp[1] >= 1 - 1e-8)
print("y =", x_lp[1], "<= 1 :", x_lp[1] <= 1 + 1e-8)
print("x =", x_lp[0], "<= 2 :", x_lp[0] <= 2 + 1e-8)
print("y =", x_lp[1], ">= 0 :", x_lp[1] >= -1e-8)

print("\n==== QP Final constraints check ====")
print("x >= 0 :", x_qp[0] >= -1e-8)
print("y >= 0 :", x_qp[1] >= -1e-8)
print("z >= 0 :", x_qp[2] >= -1e-8)
print("x + y + z = 1 :", np.abs(x_qp[0] + x_qp[1] + x_qp[2] - 1) < 1e-8)