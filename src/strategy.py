# src/strategy.py

import numpy as np
import pandas as pd
from src.bond_pricing import generate_yield_curve, zero_coupon_bond_price_vasicek

def identify_arbitrage_opportunities(
    market_yields: pd.Series,
    r_t: float,
    a: float,
    b: float,
    sigma: float,
    maturities: np.ndarray,
    threshold: float = 0.0025  # 25 basis points
):
    """
    Identify arbitrage opportunities based on yield curve mispricing.

    Parameters:
    - market_yields: pandas Series indexed by maturity
    - r_t, a, b, sigma: Vasicek parameters
    - maturities: array of maturities
    - threshold: minimum yield spread to qualify as arbitrage

    Returns:
    - DataFrame with maturities, market yield, model yield, spread, and signal
    """
    model_yields = generate_yield_curve(r_t, a, b, sigma, maturities)
    
    result = pd.DataFrame({
        'Maturity': maturities,
        'Market_Yield': market_yields.values,
        'Model_Yield': model_yields,
    })
    
    result['Spread'] = result['Market_Yield'] - result['Model_Yield']
    
    # Signal: +1 means market yield is higher (bond is underpriced)
    #         -1 means market yield is lower (bond is overpriced)
    result['Signal'] = result['Spread'].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
    
    return result
