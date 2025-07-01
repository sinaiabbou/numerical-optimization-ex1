import numpy as np

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    """
    Interior point method using log-barrier for inequality constraints and KKT for equality constraints.
    """

    mu = 10.0
    tol_outer = 1e-10
    tol_inner = 1e-8
    max_outer = 50

    x = x0.copy()
    t = 1.0
    m = len(ineq_constraints)
    history = []

    for _ in range(max_outer):
        def barrier_obj(x_inner):
            f_val, grad, hess = func(x_inner)
            barrier = 0
            barrier_grad = np.zeros_like(x_inner)
            barrier_hess = np.zeros((len(x_inner), len(x_inner)))
            for g in ineq_constraints:
                g_val, g_grad, g_hess = g(x_inner)
                if g_val <= 0:
                    return np.inf, np.zeros_like(x_inner), np.zeros((len(x_inner), len(x_inner)))
                barrier -= np.log(g_val)
                barrier_grad -= g_grad / g_val
                barrier_hess += (g_grad @ g_grad.T) / (g_val**2) - g_hess / g_val
            return t*f_val + barrier, t*grad + barrier_grad, t*hess + barrier_hess

        # Inner Newton loop
        for _ in range(100):
            f_val, grad, hess = barrier_obj(x)

            if eq_constraints_mat.size == 0:
                # Unconstrained Newton step with fallback to Gradient Descent
                try:
                    p = -np.linalg.solve(hess, grad)
                    if np.linalg.norm(p) < 1e-8:
                        raise np.linalg.LinAlgError
                except np.linalg.LinAlgError:
                    p = -grad
            else:
                # KKT system for equality constraints
                A = eq_constraints_mat
                b_eq = eq_constraints_rhs
                n, p_dim = len(x), A.shape[0]
                KKT = np.block([[hess, A.T], [A, np.zeros((p_dim, p_dim))]])
                rhs = -np.concatenate([grad, A @ x - b_eq])
                try:
                    sol = np.linalg.solve(KKT, rhs)
                    p = sol[:n]
                except np.linalg.LinAlgError:
                    return x, f_val, False, history

            # Compute alpha_max feasibility-aware always
            alpha_max = np.inf
            for g in ineq_constraints:
                g_val, g_grad, _ = g(x)
                denom = g_grad @ p
                if denom < 0:
                    alpha_i = -g_val / denom
                    if alpha_i < alpha_max:
                        alpha_max = alpha_i
            alpha_max *= 0.99

            # Backtracking line search with feasibility
            alpha = min(1.0, alpha_max)
            while True:
                x_new = x + alpha*p
                feasible = all(g(x_new)[0] > 0 for g in ineq_constraints)
                if feasible:
                    f_new, _, _ = barrier_obj(x_new)
                    if f_new < f_val + 1e-4 * alpha * grad @ p:
                        break
                alpha *= 0.5
                if alpha < 1e-10:
                    break

            x = x_new

            if np.linalg.norm(p) < tol_inner:
                break

        history.append((x.copy(), f_val))
        if m / t < tol_outer:
            return x, func(x)[0], True, history

        t *= mu

    return x, func(x)[0], False, history