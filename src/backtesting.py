import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_arbitrage_strategy(market_prices, vasicek_prices, threshold=0.02, capital=1_000_000):
    """
    Simple arbitrage backtest: Long/Short bond if mispricing exceeds threshold.
    Args:
        market_prices (pd.Series): Observed bond prices
        vasicek_prices (pd.Series): Model-implied prices
        threshold (float): % deviation threshold to enter trades
        capital (float): Initial capital

    Returns:
        pd.DataFrame: Portfolio value, returns, positions
    """
    df = pd.DataFrame({
        'market': market_prices,
        'model': vasicek_prices
    })

    df['signal'] = 0
    df['diff_pct'] = (df['market'] - df['model']) / df['model']

    # Entry rules
    df.loc[df['diff_pct'] > threshold, 'signal'] = -1  # short market
    df.loc[df['diff_pct'] < -threshold, 'signal'] = 1  # long market

    df['position'] = df['signal'].shift(1).fillna(0)
    df['market_return'] = df['market'].pct_change().fillna(0)
    df['strategy_return'] = df['position'] * df['market_return']

    df['cum_return'] = (1 + df['strategy_return']).cumprod() * capital
    df['strategy_return'] = df['strategy_return'] * 100  # % return

    return df

def plot_backtest(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['cum_return'], label='Strategy Portfolio Value')
    plt.title('Backtest Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (₹)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
