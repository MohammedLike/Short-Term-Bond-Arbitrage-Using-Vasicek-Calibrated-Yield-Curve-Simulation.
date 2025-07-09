import numpy as np

def zero_coupon_bond_price_vasicek(r_t, a, b, sigma, T):
    
    B = (1 - np.exp(-a * T)) / a
    A = np.exp((B - T) * (a**2 * b - 0.5 * sigma**2) / a**2 - (sigma**2 * B**2) / (4 * a))
    P = A * np.exp(-B * r_t)
    return P


def zero_coupon_bond_yield(price, T):
   
    return -np.log(price) / T


def generate_yield_curve(r_t, a, b, sigma, maturities):
    
    prices = [zero_coupon_bond_price_vasicek(r_t, a, b, sigma, T) for T in maturities]
    yields = [zero_coupon_bond_yield(p, T) for p, T in zip(prices, maturities)]
    return yields
