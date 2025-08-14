import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Simple Moving Average (SMA) of Close Prices
    lookback_sma = 20
    sma_close = df['close'].rolling(window=lookback_sma).mean()
    
    # Compute Volume-Adjusted Volatility
    high_low_diff = df['high'] - df['low']
    volume_adjusted_volatility = (high_low_diff * df['volume']).rolling(window=lookback_sma).mean()
    
    # Compute Price Momentum
    n = 10
    price_momentum = (df['close'] - sma_close) / df['close'].rolling(window=n).mean()
    
    # Incorporate Additional Price Change Metrics
    pct_change_lookback = 5
    percentage_change = df['close'].pct_change(periods=pct_change_lookback)
    high_low_range = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    long_term_lookback = 60
    long_term_sma = df['close'].rolling(window=long_term_lookback).mean()
    trend_indicator = np.where(sma_close > long_term_sma, 1, -1)
    
    # Calculate the Slope of the Long-Term SMA
    slope_long_term_sma = (long_term_sma - long_term_sma.shift(pct_change_lookback)) / pct_change_lookback
    adaptive_trend_strength = np.where(slope_long_term_sma > 0, 1.5, 0.5)
    trend_indicator *= adaptive_trend_strength
    
    # Incorporate Dynamic Liquidity Measures
    daily_turnover = df['volume'] * df['close']
    rolling_avg_turnover = daily_turnover.rolling(window=lookback_sma).mean()
    liquidity_weight = np.clip(rolling_avg_turnover / rolling_avg_turnover.mean(), 0, 1)
    
    # Final Alpha Factor
    weights = {
        'price_momentum': 0.4,
        'volume_adjusted_volatility': -0.2,
        'percentage_change': 0.3,
        'liquidity_weight': 0.1
    }
    
    alpha_factor = (
        weights['price_momentum'] * price_momentum +
        weights['volume_adjusted_volatility'] * volume_adjusted_volatility +
        weights['percentage_change'] * percentage_change +
        weights['liquidity_weight'] * liquidity_weight
    )
    
    # Dynamically Adjust Weights Based on Market Trend
    alpha_factor *= np.where(trend_indicator > 0, 1.2, 0.8)
    
    return alpha_factor
