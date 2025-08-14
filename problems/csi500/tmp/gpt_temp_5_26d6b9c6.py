import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df, short_sma_period=20, long_sma_period=200, vol_lookback=20, momentum_lookback=60, pct_change_lookback=5, turnover_lookback=20):
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_short'] = df['close'].rolling(window=short_sma_period).mean()
    df['SMA_long'] = df['close'].rolling(window=long_sma_period).mean()

    # Compute Volume-Adjusted Volatility
    df['high_low_diff'] = df['high'] - df['low']
    df['vol_weighted_vol'] = df['high_low_diff'] * df['volume']
    df['vol_adj_vol'] = df['vol_weighted_vol'].rolling(window=vol_lookback).mean()

    # Compute Price Momentum
    df['price_momentum'] = (df['close'] - df['SMA_short']) / df['close'].rolling(window=momentum_lookback).mean()

    # Incorporate Additional Price Change Metrics
    df['pct_change'] = df['close'].pct_change(periods=pct_change_lookback)
    df['high_low_range'] = df['high'] - df['low']

    # Consider Market Trend Alignment
    df['trend_indicator'] = (df['SMA_short'] > df['SMA_long']).astype(int)

    # Incorporate Liquidity Measures
    df['daily_turnover'] = df['volume'] * df['close']
    df['avg_turnover'] = df['daily_turnover'].rolling(window=turnover_lookback).mean()

    # Machine Learning for Dynamic Weight Adjustments and Feature Selection
    features = ['price_momentum', 'vol_adj_vol', 'pct_change', 'high_low_range', 'trend_indicator', 'avg_turnover']
    X = df[features].dropna()
    y = df['close'].shift(-1).loc[X.index]  # Predict future returns

    model = LinearRegression()
    model.fit(X, y)
    weights = model.coef_ / model.coef_.sum()  # Ensure the weights sum to 1

    # Adjust Weights Based on Market Trend
    bullish_weights = weights * 1.2 if df['trend_indicator'].iloc[-1] == 1 else weights * 0.8

    # Adjust Weights Based on Liquidity
    liquidity_adjusted_weights = bullish_weights * (df['avg_turnover'] / df['avg_turnover'].mean())

    # Final Alpha Factor
    df['alpha_factor'] = (X * liquidity_adjusted_weights).sum(axis=1)

    return df['alpha_factor'].dropna()
