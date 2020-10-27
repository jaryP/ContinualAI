import numpy as np
import quadprog


def qp(past_tasks_gradient, current_gradient, margin):
    t = past_tasks_gradient.shape[0]
    P = np.dot(past_tasks_gradient, past_tasks_gradient.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
    q = np.dot(past_tasks_gradient, current_gradient) * -1
    q = np.squeeze(q, 1)
    h = np.zeros(t) + margin
    G = np.eye(t)
    v = quadprog.solve_qp(P, q, G, h)[0]
    return v
