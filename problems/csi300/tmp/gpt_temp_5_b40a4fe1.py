import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Adjusted High-Low Momentum
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Compute Rolling Volatility (20-day standard deviation of returns)
    returns = df['close'].pct_change()
    rolling_volatility = returns.rolling(window=20).std()
    
    # Combine Range and Volatility
    volatility_adjusted_range = high_low_range / (rolling_volatility + 1e-8)
    
    # Apply sign from price momentum (5-day price change)
    price_momentum = df['close'].pct_change(5)
    volatility_momentum = volatility_adjusted_range * np.sign(price_momentum)
    
    # Volume-Price Divergence Factor
    # Calculate Price Trend
    price_ema_10 = df['close'].ewm(span=10).mean()
    price_ema_5 = df['close'].ewm(span=5).mean()
    
    # Calculate Volume Trend
    volume_ema_10 = df['volume'].ewm(span=10).mean()
    volume_ema_5 = df['volume'].ewm(span=5).mean()
    
    # Compute Divergence
    price_slope = (price_ema_5 - price_ema_10) / price_ema_10
    volume_slope = (volume_ema_5 - volume_ema_10) / volume_ema_10
    volume_price_divergence = price_slope * volume_slope
    
    # Intraday Reversal Strength
    # Calculate Opening Gap
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Intraday Recovery
    gap_up_recovery = np.where(opening_gap > 0, 
                              (df['high'] - df['open']) / (df['open'] - df['low'] + 1e-8), 
                              (df['low'] - df['open']) / (df['open'] - df['high'] + 1e-8))
    
    # Weight by Historical Pattern
    gap_threshold = 0.01
    similar_gaps = opening_gap.rolling(window=60).apply(
        lambda x: np.mean(np.abs(x[1:]) > gap_threshold) if len(x) > 1 else 0
    )
    reversal_strength = gap_up_recovery * similar_gaps
    
    # Liquidity-Efficient Price Movement
    # Compute Price Efficiency
    price_change = df['close'].pct_change().abs()
    volume_efficiency = price_change / (df['volume'] + 1e-8)
    
    # Calculate Movement Quality
    intraday_range = (df['high'] - df['low']) / df['close']
    movement_consistency = (df['close'] - df['open']).abs() / (intraday_range + 1e-8)
    movement_quality = volume_efficiency * movement_consistency
    
    # Cumulative Pressure Indicator
    # Track Accumulation/Distribution
    daily_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    pressure_volume = daily_position * df['volume']
    
    # Calculate Pressure Building
    cumulative_pressure = pressure_volume.rolling(window=10).sum()
    pressure_percentile = cumulative_pressure.rolling(window=60).apply(
        lambda x: (x[-1] > np.percentile(x[:-1], 80)) if len(x) > 1 else 0
    )
    
    # Combine all factors
    factor = (volatility_momentum * 0.3 + 
              volume_price_divergence * 0.25 + 
              reversal_strength * 0.2 + 
              movement_quality * 0.15 + 
              pressure_percentile * 0.1)
    
    return factor
