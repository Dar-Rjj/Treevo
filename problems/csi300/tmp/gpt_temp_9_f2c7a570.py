import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Decay Component
    # Calculate daily returns from close prices
    returns = df['close'].pct_change()
    
    # Apply exponential weighting over 10-day window (decay=0.9)
    ewma_returns = returns.ewm(span=10, alpha=0.1).mean()
    
    # Compute momentum acceleration with 2-day lag
    momentum_acceleration = ewma_returns - ewma_returns.shift(2)
    
    # Volume Divergence Component
    # Compute volume anomaly detection using z-score relative to 20-day history
    volume_mean = df['volume'].rolling(window=20).mean()
    volume_std = df['volume'].rolling(window=20).std()
    volume_anomaly = (df['volume'] - volume_mean) / volume_std
    
    # Measure price-volume divergence
    # Calculate rolling correlation between returns and volume (15-day window)
    rolling_corr = returns.rolling(window=15).corr(df['volume'])
    # Compute divergence as absolute deviation from historical average correlation
    avg_correlation = rolling_corr.rolling(window=20).mean()
    divergence = abs(rolling_corr - avg_correlation)
    
    # Volatility Scaling Component
    # Calculate daily true range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate 10-day moving average of true range
    volatility = true_range.rolling(window=10).mean()
    
    # Incorporate volatility regime
    vol_20day_high = volatility.rolling(window=20).max()
    vol_20day_low = volatility.rolling(window=20).min()
    vol_regime = (volatility - vol_20day_low) / (vol_20day_high - vol_20day_low)
    
    # Create volatility adjustment factor based on regime
    volatility_adjustment = 1 + vol_regime
    
    # Composite Factor Generation
    # Combine momentum with volume confirmation
    momentum_volume_component = momentum_acceleration * volume_anomaly * (1 + divergence)
    
    # Apply volatility scaling
    final_factor = momentum_volume_component / (volatility * volatility_adjustment)
    
    return final_factor
