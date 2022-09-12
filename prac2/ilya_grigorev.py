import numpy as np
from scipy.signal import convolve, correlate
from scipy.stats import norm


def calculate_log_probability(X, F, B, s):
    X_prod_B = X * B[..., np.newaxis]
    B_2 = B ** 2
    eps = 1e-10
    norm_pdf = norm.pdf(X, 0, s)
    norm_pdf[norm_pdf < eps] = eps
    res = np.sum(np.log(norm_pdf), axis=(0, 1)) + \
          (2 * (correlate(X, F[..., np.newaxis], mode='valid') +
                np.sum(X_prod_B, axis=(0, 1))) - np.sum(F ** 2) - np.sum(B_2)) / (2 * (s ** 2))

    for m in range(X.shape[0] - F.shape[0] + 1):
        for n in range(X.shape[1] - F.shape[1] + 1):
            res[m, n] += (np.sum(B_2[m : m + F.shape[0], n : n + F.shape[1]]) -
                          2 * np.sum(X_prod_B[m : m + F.shape[0], n : n + F.shape[1]], axis=(0, 1))) / (2 * (s ** 2))
    return res


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    log_probability = calculate_log_probability(X, F, B, s)
    eps = 1e-10
    A_log = A.copy()
    A_log[A_log < eps] = eps
    q_log = q.copy()
    q_log[q_log < eps] = eps
    if use_MAP:
        return (log_probability[q[0], q[1], np.arange(X.shape[2])]).sum() + (np.log(A_log)[q[0], q[1]]).sum()
    else:
        return (q * log_probability).sum() + (np.sum(q, axis=2) * np.log(A_log)).sum() - (q * np.log(q_log)).sum()


def run_e_step(X, F, B, s, A, use_MAP=False):
    log_probability = calculate_log_probability(X, F, B, s)
    eps = 1e-10
    A_log = A.copy()
    A_log[A_log < eps] = eps
    if use_MAP:
        q = np.log(A_log)[..., np.newaxis] + log_probability
        q -= q.max(axis=(0, 1))
        q = np.exp(q - np.log(np.exp(q).sum(axis=(0, 1))))
        num_w = q.shape[1]
        q = q.reshape((-1, q.shape[2])).argmax(axis=0)
        return np.vstack((q // num_w, q % num_w))
    else:
        q = np.log(A_log)[..., np.newaxis] + log_probability
        q -= q.max(axis=(0, 1))
        return np.exp(q - np.log(np.sum(np.exp(q), axis=(0, 1))))


def run_m_step(X, q, h, w, use_MAP=False):
    eps = 1e-10
    if use_MAP:
        A = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1, X.shape[2]))
        A[q[0], q[1], np.arange(X.shape[2])] = 1.0
        A = np.mean(A, axis=2)

        row_idx = np.tile(np.arange(h)[..., np.newaxis], (1, w))
        row_idx = row_idx[..., np.newaxis] + q[0].reshape((1, 1, -1))
        column_idx = np.tile(np.arange(w)[np.newaxis, ...], (h, 1))
        column_idx = column_idx[..., np.newaxis] + q[1].reshape((1, 1, -1))
        img_idx = np.tile(np.arange(X.shape[2]).reshape((1, 1, -1)), (h, w, 1))
        F = X[row_idx, column_idx, img_idx].mean(axis=2)

        row_limits = np.tile(np.arange(X.shape[0])[..., np.newaxis], (1, X.shape[2]))
        row_mask = ~((q[0][np.newaxis, ...] <= row_limits) * (row_limits <= q[0][np.newaxis, ...] + h - 1))
        column_limits = np.tile(np.arange(X.shape[1])[..., np.newaxis], (1, X.shape[2]))
        column_mask = ~((q[1][np.newaxis, ...] <= column_limits) * (column_limits <= q[1][np.newaxis, ...] + w - 1))
        mask = row_mask[:, np.newaxis, :] + column_mask[np.newaxis, ...]
        B = np.sum(X * mask, axis=2) / (np.sum(mask, axis=2) + eps)

        s = 0.0
        for k in range(X.shape[2]):
            temp = np.copy(B)
            temp[q[0, k] : q[0, k] + h, q[1, k] : q[1, k] + w] = F
            s += np.sum((X[:, :, k] - temp) ** 2)
        s = np.sqrt(s / (X.shape[0] * X.shape[1] * X.shape[2]))
        return F, B, s, A
    else:
        A = np.mean(q, axis=2)

        F_num = correlate(X, q, mode='valid').squeeze(2)
        F_num[F_num < eps] = eps
        F = np.exp(np.log(F_num) - np.log(q.sum() + eps))

        q_sum = np.sum(q, axis=(0, 1)) - convolve(q, np.ones((h, w, 1)))
        B_num = np.sum(X * q_sum, axis=2)
        B_den = np.sum(q_sum, axis=2)
        B_num[B_num < eps] = eps
        B_den[B_den < eps] = eps
        B = np.exp(np.log(B_num) - np.log(B_den))

        s = 0.0
        for m in range(X.shape[0] - h + 1):
            for n in range(X.shape[1] - w + 1):
                temp = np.copy(B)
                temp[m : m + h, n : n + w] = F
                s += np.sum(q[m, n] * np.sum((X - temp[..., np.newaxis]) ** 2, axis=(0, 1)))
        s = np.sqrt(s / (X.shape[0] * X.shape[1] * X.shape[2]))
        return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001, max_iter=50, use_MAP=False):
    if F is None:
        F = np.random.uniform(0, 255, (h, w))
    if B is None:
        B = np.random.uniform(0, 255, (X.shape[0], X.shape[1]))
    if s is None:
        s = 1e-1
    if A is None:
        A = np.random.rand(X.shape[0] - h + 1, X.shape[1] - w + 1)
        A /= A.sum()

    q = run_e_step(X, F, B, s, A, use_MAP)
    F_max, B_max, s_max, A_max = run_m_step(X, q, h, w, use_MAP)
    lower_bound_new = calculate_lower_bound(X, F_max, B_max, s_max, A_max, q, use_MAP)
    lower_bound_list = np.array([lower_bound_new])
    for i in range(max_iter - 1):
        # print(f'Iter: {i}')
        # print(f'ELBO: {lower_bound_new}')
        lower_bound_old = lower_bound_new
        q = run_e_step(X, F_max, B_max, s_max, A_max, use_MAP)
        F_max, B_max, s_max, A_max = run_m_step(X, q, h, w, use_MAP)
        lower_bound_new = calculate_lower_bound(X, F_max, B_max, s_max, A_max, q, use_MAP)
        lower_bound_list = np.append(lower_bound_list, lower_bound_new)
        if abs(lower_bound_new - lower_bound_old) < tolerance:
            break
    # else:
    #     print('Warning convergence may not achieved')
    return F_max, B_max, s_max, A_max, lower_bound_list


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False, n_restarts=10):
    F = np.random.rand(h, w)
    B = np.random.rand(X.shape[0], X.shape[1])
    s = np.random.uniform(1e-3, 1e-1)
    A = np.random.rand(X.shape[0] - h + 1, X.shape[1] - w + 1)
    A /= A.sum()
    if n_restarts > 0:
        F, B, s, A, lower_bound = run_EM(X, h, w, F, B, s, A, tolerance, max_iter, use_MAP)
        lower_bound_max = lower_bound[-1]
        F_res = F
        B_res = B
        s_res = s
        A_res = A
        for i in range(n_restarts - 1):
            F = np.random.rand(h, w)
            B = np.random.rand(X.shape[0], X.shape[1])
            s = np.random.uniform(1e-3, 1e-1)
            A = np.random.rand(X.shape[0] - h + 1, X.shape[1] - w + 1)
            A /= A.sum()
            F, B, s, A, lower_bound = run_EM(X, h, w, F, B, s, A, tolerance, max_iter, use_MAP)
            if lower_bound[-1] > lower_bound_max:
                lower_bound_max = lower_bound[-1]
                F_res = F
                B_res = B
                s_res = s
                A_res = A
        return F_res, B_res, s_res, A_res, lower_bound_max
