# ----------------------------------------------------------------------
# objective function for corr
# ----------------------------------------------------------------------

# Note: if we wish to use this objective function with LightGBM, we need 
# to take into account the fact that Gradient Boosted Decision Trees 
# start by considering vector with all entries equal to the average of 
# y_true. Such vector has 0 standard deviation, and grad/hess are not 
# defined. We can fix this by training some random tree and feeding that
# tree to LGBM as init_model
def objective_corr(y_true, y_pred):
    m = len(y_true)

    var_e = (m - 1) / m**2
    std_t = np.std(y_true)
    std_p = np.std(y_pred)

    cov_tp = np.cov(y_true, y_pred, ddof=0)[0, 1]
    cov_ep = (y_pred - np.mean(y_pred)) / m
    cov_et = (y_true - np.mean(y_true)) / m

    grad = - (- cov_tp * cov_ep / std_p**3 + cov_et / std_p) / std_t
    hess = - (- 2 * cov_ep * cov_et
              + 3 * cov_ep**2 * cov_tp / std_p**2
              - cov_tp * var_e) / (std_t * std_p**3)

    return grad, hess


def objective_corr_ones(y_true, y_pred):
    m = len(y_true)

    std_t = np.std(y_true)
    std_p = np.std(y_pred)

    cov_tp = np.cov(y_true, y_pred, ddof=0)[0, 1]
    cov_ep = (y_pred - np.mean(y_pred)) / m
    cov_et = (y_true - np.mean(y_true)) / m

    grad = - (- cov_tp * cov_ep / std_p**3 + cov_et / std_p) / std_t
    hess = np.ones(m)

    return grad, hess


# objective function for corr, numeric (just to test that objective_corr
# is correct)
def objective_corr_num(y_true, y_pred):
    t = y_true
    p = y_pred
    h = 10 ** (-8)
    m = len(t)
    I = np.eye(m)

    corr_zero = corr(t, p)
    corr_plus = corr(t, (p + h*I).T)
    corr_minu = corr(t, (p - h*I).T)

    grad = - (corr_plus - corr_zero) / h
    hess = - (corr_plus + corr_minu - 2*corr_zero) / h**2

    return grad, hess


# ======================================================================
# utils
# ======================================================================


def now_dt():
    return datetime.now().strftime('%Y-%m-%d-%H-%M')


def maximum(f, n, k=0.01, n_iters=10000):
    def w_new(i_dec, i_inc, w):
        w_ret = np.copy(w)
        w_ret[i_dec] -= k
        w_ret[i_inc] += k
        return w_ret

    w = np.ones(n) / n

    for _ in range(n_iters):
        pairs = product(range(n), range(n))
        values = [(f(w_new(i_dec, i_inc, w)), i_dec, i_inc)
                  for i_dec, i_inc in pairs]
        values = sorted(values, reverse=True)
        _, i_dec, i_inc = values[0]
        if i_dec == i_inc:
            break
        w = w_new(i_dec, i_inc, w)

    return w