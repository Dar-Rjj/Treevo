import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Efficiency
    high_low_range = df['high'] - df['low']
    high_low_range = high_low_range.replace(0, np.nan)  # Avoid division by zero
    intraday_efficiency = (df['close'] - df['low']) / high_low_range
    
    # Calculate Short-term Price Acceleration
    roc_5 = df['close'].pct_change(5)
    roc_10 = df['close'].pct_change(10)
    price_acceleration = roc_5 - roc_10
    
    # Calculate Realized Volatility
    returns = df['close'].pct_change()
    realized_vol = returns.rolling(window=20, min_periods=10).std()
    
    # Classify Volatility Regime
    vol_percentile = realized_vol.rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 70)) * 2 + 
                 (x.iloc[-1] < np.percentile(x.dropna(), 30)) * -1 if len(x.dropna()) >= 10 else 0, 
        raw=False
    )
    
    # Calculate Volume-Weighted Price Position
    volume_weighted_position = (df['close'] - df['low']) / high_low_range * df['volume']
    volume_weighted_position = volume_weighted_position / df['volume']  # Normalize by volume
    
    # Assess Volume-Price Alignment
    volume_ratio = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    volume_confirmation = np.sign(price_acceleration) * np.sign(volume_ratio - 1)
    volume_strength = np.abs(volume_ratio - 1) * np.abs(volume_confirmation)
    
    # Combine Intraday Momentum Components
    intraday_momentum = intraday_efficiency * price_acceleration
    
    # Apply Volatility-Dependent Scaling
    volatility_weights = np.where(vol_percentile == -1, 1.5,  # Low volatility: amplify
                         np.where(vol_percentile == 2, 0.5,   # High volatility: dampen
                                 1.0))                       # Normal volatility: neutral
    
    # Adjust by Volume Synchronization Strength
    volume_adjustment = 1 + 0.5 * volume_strength
    
    # Final Factor Calculation
    factor = intraday_momentum * volatility_weights * volume_adjustment
    
    return factor
