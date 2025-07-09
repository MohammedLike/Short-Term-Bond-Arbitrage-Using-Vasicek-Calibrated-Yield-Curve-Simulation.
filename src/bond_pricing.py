# src/bond_pricing.py

import numpy as np

def zero_coupon_bond_price_vasicek(r_t, a, b, sigma, T):
    """
    Price of a zero-coupon bond under the Vasicek model.

    Parameters:
    - r_t: current short rate
    - a: speed of mean reversion
    - b: long-term mean
    - sigma: volatility
    - T: time to maturity (in years)

    Returns:
    - Zero-coupon bond price at time t for maturity T
    """
    B = (1 - np.exp(-a * T)) / a
    A = np.exp((B - T) * (a**2 * b - 0.5 * sigma**2) / a**2 - (sigma**2 * B**2) / (4 * a))
    P = A * np.exp(-B * r_t)
    return P


def zero_coupon_bond_yield(price, T):
    """
    Compute the continuously compounded yield from bond price.

    Parameters:
    - price: bond price
    - T: time to maturity

    Returns:
    - yield (continuously compounded)
    """
    return -np.log(price) / T


def generate_yield_curve(r_t, a, b, sigma, maturities):
    """
    Generate Vasicek-implied yield curve.

    Parameters:
    - r_t: current short rate
    - a, b, sigma: Vasicek parameters
    - maturities: array of maturities (in years)

    Returns:
    - yields: list of yields corresponding to each maturity
    """
    prices = [zero_coupon_bond_price_vasicek(r_t, a, b, sigma, T) for T in maturities]
    yields = [zero_coupon_bond_yield(p, T) for p, T in zip(prices, maturities)]
    return yields
