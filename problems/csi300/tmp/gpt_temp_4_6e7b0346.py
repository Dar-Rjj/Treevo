import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Smoothed Multi-Timeframe Trend Alignment Factor
    Combines long-term (20-day) and medium-term (10-day) price and volume trends,
    adjusted by smoothed volatility measures to create a robust alpha factor.
    """
    # Extract necessary columns
    close = df['close']
    volume = df['volume']
    
    # Long-Term Trend Component (20-day)
    long_price_mean = close.shift(1).rolling(window=20, min_periods=10).mean()
    long_price_strength = (close - long_price_mean) / long_price_mean
    
    long_volume_mean = volume.shift(1).rolling(window=20, min_periods=10).mean()
    long_volume_strength = (volume - long_volume_mean) / long_volume_mean
    
    # Medium-Term Trend Component (10-day)
    medium_price_mean = close.shift(1).rolling(window=10, min_periods=5).mean()
    medium_price_strength = (close - medium_price_mean) / medium_price_mean
    
    medium_volume_mean = volume.shift(1).rolling(window=10, min_periods=5).mean()
    medium_volume_strength = (volume - medium_volume_mean) / medium_volume_mean
    
    # Smoothed Volatility Assessment
    # Price volatility using Mean Absolute Deviation of daily returns
    daily_returns = close.pct_change().shift(1)
    price_volatility = daily_returns.rolling(window=20, min_periods=10).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    )
    
    # Volume volatility using Mean Absolute Deviation of daily volume changes
    daily_volume_changes = volume.pct_change().shift(1)
    volume_volatility = daily_volume_changes.rolling(window=20, min_periods=10).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    )
    
    # Multi-Timeframe Signal Integration
    long_term_alignment = long_price_strength * long_volume_strength
    medium_term_alignment = medium_price_strength * medium_volume_strength
    
    combined_trend_alignment = long_term_alignment * medium_term_alignment
    
    # Volatility-Adjusted Final Signal
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    price_volatility_adj = price_volatility.replace(0, epsilon)
    volume_volatility_adj = volume_volatility.replace(0, epsilon)
    
    volatility_adjusted_signal = combined_trend_alignment / price_volatility_adj
    final_factor = volatility_adjusted_signal / volume_volatility_adj
    
    return final_factor
