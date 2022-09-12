import numpy as np
from scipy.stats import binom, poisson


# There should be no main() in this file!!!
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1,
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,)
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

# In variant 2 the following functions are required:
def pa(params, model):
    val = np.arange(params['amin'], params['amax'] + 1)
    prob = np.full(val.shape[0], 1 / val.shape[0])
    return prob, val


def pb(params, model):
    val = np.arange(params['bmin'], params['bmax'] + 1)
    prob = np.full(val.shape[0], 1 / val.shape[0])
    return prob, val


def pc(params, model):
    if model == 1:
        prob_a, val_a = pa(params, model)
        prob_b, val_b = pb(params, model)
        binom_a = binom.pmf(np.arange(params['amax'] + 1)[:, np.newaxis],
                            np.arange(params['amin'], params['amax'] + 1)[np.newaxis, :], params['p1'])
        binom_b = binom.pmf(np.arange(params['bmax'] + 1)[:, np.newaxis],
                            np.arange(params['bmin'], params['bmax'] + 1)[np.newaxis, :], params['p2'])
        binom_a_prob = np.sum(binom_a, axis=1) * prob_a[0]
        binom_b_prob = np.sum(binom_b, axis=1) * prob_b[0]
        binom_a_val = np.arange(params['amax'] + 1)
        binom_b_val = np.arange(params['bmax'] + 1)
        binom_c_val = np.ravel(binom_a_val[np.newaxis, :] + binom_b_val[:, np.newaxis])
        binom_c_prob = np.ravel(binom_a_prob[np.newaxis, :] * binom_b_prob[:, np.newaxis])
        binom_c_val_unique = np.unique(binom_c_val)
        prob_c = []
        for elem in binom_c_val_unique:
            prob_c.append(np.sum(binom_c_prob[binom_c_val == elem]))
        return np.array(prob_c), binom_c_val_unique
    elif model == 2:
        prob_a, val_a = pa(params, model)
        prob_b, val_b = pb(params, model)
        val_a = val_a * params['p1']
        val_b = val_b * params['p2']
        param_val, param_counts = np.unique(np.ravel(val_a[np.newaxis, :] + val_b[:, np.newaxis]), return_counts=True)
        param_prob = prob_a[0] * prob_b[0] * param_counts
        val_c = np.arange(params['amax'] + params['bmax'] + 1)
        prob_c = np.sum(poisson.pmf(val_c[:, np.newaxis], param_val[np.newaxis, :]) * param_prob[np.newaxis, :], axis=1)
        return prob_c, val_c


def pd(params, model):
    if model == 1 or model == 2:
        prob_c, val_c = pc(params, model)
        binom_c_p3 = binom.pmf(val_c[:, np.newaxis], val_c[np.newaxis, :], params['p3']) * prob_c[np.newaxis, :]
        val_d = np.arange(2 * (params['amax'] + params['bmax']) + 1)
        val_d_matrix = val_c[np.newaxis, :] + val_c[:, np.newaxis]
        prob_d = np.bincount(val_d_matrix.ravel(), weights=binom_c_p3.ravel())
        return prob_d, val_d


def pc_a(a, params, model):
    if model == 1:
        prob_b, val_b = pb(params, model)
        binom_a_val = np.arange(params['amax'] + 1)
        binom_a_prob = binom.pmf(binom_a_val[:, np.newaxis], a[np.newaxis, :], params['p1']).T
        binom_b_val = np.arange(params['bmax'] + 1)
        binom_b_prob = binom.pmf(binom_b_val[:, np.newaxis], val_b[np.newaxis, :], params['p2'])
        binom_b_prob = np.sum(binom_b_prob, axis=1) * prob_b[0]
        val_c = np.ravel(binom_a_val[np.newaxis, :] + binom_b_val[:, np.newaxis])
        val_c_unique = np.arange(params['amax'] + params['bmax'] + 1)
        prob_c = (binom_a_prob[:, np.newaxis, :] * binom_b_prob[np.newaxis, :, np.newaxis]).reshape(a.shape[0], -1)
        prob_res = np.apply_along_axis(lambda x: np.bincount(val_c, weights=x), 1, prob_c)
        return prob_res.T, val_c_unique
    elif model == 2:
        prob_b, val_b = pb(params, model)
        a = a * params['p1']
        val_b = val_b * params['p2']
        param_val = a[:, np.newaxis] + val_b[np.newaxis, :]
        val_c = np.arange(params['amax'] + params['bmax'] + 1)
        prob_c = np.sum(poisson.pmf(val_c[:, np.newaxis], param_val.ravel()[np.newaxis, :]).reshape(val_c.shape[0], a.shape[0], -1) * prob_b[0], axis=2)
        return prob_c, val_c


def pc_b(b, params, model):
    if model == 1:
        prob_a, val_a = pa(params, model)
        binom_b_val = np.arange(params['bmax'] + 1)
        binom_b_prob = binom.pmf(binom_b_val[:, np.newaxis], b[np.newaxis, :], params['p2']).T
        binom_a_val = np.arange(params['amax'] + 1)
        binom_a_prob = binom.pmf(binom_a_val[:, np.newaxis], val_a[np.newaxis, :], params['p1'])
        binom_a_prob = np.sum(binom_a_prob, axis=1) * prob_a[0]
        val_c = np.ravel(binom_b_val[np.newaxis, :] + binom_a_val[:, np.newaxis])
        val_c_unique = np.arange(params['amax'] + params['bmax'] + 1)
        prob_c = (binom_b_prob[:, np.newaxis, :] * binom_a_prob[np.newaxis, :, np.newaxis]).reshape(b.shape[0], -1)
        prob_res = np.apply_along_axis(lambda x: np.bincount(val_c, weights=x), 1, prob_c)
        return prob_res.T, val_c_unique
    elif model == 2:
        prob_a, val_a = pa(params, model)
        b = b * params['p2']
        val_a = val_a * params['p1']
        param_val = b[:, np.newaxis] + val_a[np.newaxis, :]
        val_c = np.arange(params['amax'] + params['bmax'] + 1, dtype=np.int16)
        prob_c = np.sum(poisson.pmf(val_c[:, np.newaxis], param_val.ravel()[np.newaxis, :]).reshape(val_c.shape[0], b.shape[0], -1) * prob_a[0], axis=2)
        return prob_c, val_c


def pb_a(a, params, model):
    prob_b, val_b = pb(params, model)
    return np.tile(prob_b, a.shape[0]).reshape(a.shape[0], -1).T, val_b


def pd_b(b, params, model):
    if model == 1 or model == 2:
        prob_c, val_c = pc_b(b, params, model)
        prob_c = prob_c.T
        val_d = np.arange(2 * (params['amax'] + params['bmax']) + 1, dtype=np.int16)
        val_d_matrix = np.ravel(val_c[np.newaxis, :] + val_c[:, np.newaxis])
        binom_c_p3_b = (binom.pmf(val_c[:, np.newaxis], val_c[np.newaxis, :], params['p3'])[np.newaxis, ...] * prob_c[:, np.newaxis, :]).reshape(b.shape[0], -1)
        prob_res = np.apply_along_axis(lambda x: np.bincount(val_d_matrix, weights=x), 1, binom_c_p3_b)
        return prob_res.T, val_d


def pb_d(d, params, model):
    eps = 1e-10
    d = d.astype(np.int16)
    if model == 1 or model == 2:
        val_b = np.arange(params['bmin'], params['bmax'] + 1, dtype=np.int16)
        prob_d_b, val_d = pd_b(val_b, params, model)
        dev = np.sum(prob_d_b, axis=1)
        dev[dev < eps] = eps
        prob_d_b = prob_d_b / dev[:, np.newaxis]
        return prob_d_b[d].T, val_b


def pc_ab(a, b, params, model):
    if model == 1:
        binom_a_val = np.arange(params['amax'] + 1)
        binom_b_val = np.arange(params['bmax'] + 1)
        binom_a_probs = binom.pmf(binom_a_val[:, np.newaxis], a[np.newaxis, :], params['p1']).astype(np.float32).T
        binom_b_probs = binom.pmf(binom_b_val[:, np.newaxis], b[np.newaxis, :], params['p2']).astype(np.float32).T
        val_c_matrix = np.ravel(binom_a_val[:, np.newaxis] + binom_b_val[np.newaxis, :])
        val_c = np.arange(params['amax'] + params['bmax'] + 1)
        prob_c = (binom_b_probs[:, np.newaxis, np.newaxis, :] * binom_a_probs[np.newaxis, :, :, np.newaxis]).reshape(b.shape[0] * a.shape[0], -1)
        prob_res = np.apply_along_axis(lambda x: np.bincount(val_c_matrix, weights=x), 1, prob_c).reshape(b.shape[0], a.shape[0], -1).T
        return prob_res, val_c
    elif model == 2:
        a = a * params['p1']
        b = b * params['p2']
        param_val = np.ravel(a[:, np.newaxis] + b[np.newaxis, :])
        val_c = np.arange(params['amax'] + params['bmax'] + 1, dtype=np.int16)
        prob_c = poisson.pmf(val_c[:, np.newaxis], param_val[np.newaxis, :]).reshape(val_c.shape[0], a.shape[0], -1)
        return prob_c, val_c


def pd_ab(a, b, params, model):
    if model == 1 or model == 2:
        prob_c_ab, val_c = pc_ab(a, b, params, model)
        prob_c_ab = prob_c_ab.astype(np.float32).T
        val_d = np.arange(2 * (params['amax'] + params['bmax']) + 1, dtype=np.int16)
        val_d_matrix = np.ravel(val_c[np.newaxis, :] + val_c[:, np.newaxis])
        binom_c_p3_ab = (binom.pmf(val_c[:, np.newaxis], val_c[np.newaxis, :], params['p3']).astype(np.float32)[np.newaxis, np.newaxis, ...] * prob_c_ab[:, :, np.newaxis, :]).reshape(b.shape[0] * a.shape[0], -1)
        prob_res = np.apply_along_axis(lambda x: np.bincount(val_d_matrix, weights=x), 1, binom_c_p3_ab).astype(np.float32).reshape(b.shape[0], a.shape[0], -1).T
        return prob_res, val_d


def pb_ad(a, d, params, model):
    eps = 1e-10
    d = d.astype(np.int16)
    if model == 1 or model == 2:
        val_b = np.arange(params['bmin'], params['bmax'] + 1, dtype=np.int16)
        prob_d_ab, val_d = pd_ab(a, val_b, params, model)
        dev = np.sum(prob_d_ab, axis=2)
        dev[dev < eps] = eps
        prob_d_ab = prob_d_ab / dev[..., np.newaxis]
        return prob_d_ab[d].T, val_b
