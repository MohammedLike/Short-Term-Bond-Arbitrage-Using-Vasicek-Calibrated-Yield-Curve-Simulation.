import numpy as np
from scipy.optimize import minimize
from math import exp


def calibrate_vasicek(rates, dt=1/252):
   
    r = rates.values
    r_t = r[:-1]
    r_tp1 = r[1:]

    def neg_log_likelihood(params):
        a, b, sigma = params
        if sigma <= 0 or a <= 0:
            return np.inf
        mu = r_t + a * (b - r_t) * dt
        var = sigma ** 2 * dt
        loglik = -0.5 * np.sum(np.log(2 * np.pi * var) + ((r_tp1 - mu) ** 2) / var)
        return -loglik

    initial_guess = [0.1, np.mean(r), 0.01]
    bounds = [(1e-6, 5), (0, 1), (1e-6, 1)]

    result = minimize(neg_log_likelihood, initial_guess, bounds=bounds)

    if result.success:
        return result.x  # a, b, sigma
    else:
        raise ValueError("Vasicek calibration failed.")



def simulate_vasicek_paths(r0, a, b, sigma, T=1.0, n_steps=252, n_paths=10, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    rates = np.zeros((n_steps + 1, n_paths))
    rates[0] = r0

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        rates[t] = rates[t - 1] + a * (b - rates[t - 1]) * dt + sigma * np.sqrt(dt) * z

    return rates



def zero_coupon_bond_price(r, a, b, sigma, T):
   
    B = (1 - exp(-a * T)) / a
    A = exp((B - T) * (a**2 * b - 0.5 * sigma**2) / (a**2) - (sigma**2 * B**2) / (4 * a))
    return A * exp(-B * r)
