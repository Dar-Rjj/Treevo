import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volume Divergence with Decay Weighting factor
    
    Combines price momentum acceleration with volume-price alignment,
    applying temporal decay and volatility scaling for robust signal generation.
    """
    close = df['close']
    volume = df['volume']
    
    # Price Momentum Component
    # Exponential Weighted Momentum (20-day, decay=0.9)
    momentum = close.ewm(span=20, adjust=False).mean()
    
    # Momentum Acceleration (first derivative)
    momentum_acceleration = momentum.diff()
    
    # Volume-Price Alignment
    # Daily returns
    returns = close.pct_change()
    
    # Volume-weighted price changes (cube root preserves direction, reduces skew)
    volume_weighted_changes = returns * np.cbrt(volume)
    
    # Rolling correlation between momentum and volume-weighted changes (10-day)
    momentum_series = momentum_acceleration.rolling(window=10, min_periods=5).mean()
    volume_series = volume_weighted_changes.rolling(window=10, min_periods=5).mean()
    
    # Calculate rolling correlation
    rolling_corr = momentum_series.rolling(window=10, min_periods=5).corr(volume_series)
    
    # Signal Integration with Temporal Decay
    # Apply exponential decay weighting (factor=0.95)
    decay_weights = pd.Series(0.95 ** np.arange(len(close)), index=close.index)
    decay_weights = decay_weights / decay_weights.sum()  # Normalize
    
    # Combine momentum acceleration and correlation with decay weighting
    combined_signal = (momentum_acceleration * rolling_corr).rolling(
        window=len(decay_weights), min_periods=10
    ).apply(lambda x: np.sum(x * decay_weights.iloc[:len(x)]), raw=False)
    
    # Volatility Scaling
    # 20-day rolling volatility (standard deviation of returns)
    volatility = returns.rolling(window=20, min_periods=10).std()
    
    # Scale final factor by volatility
    final_factor = combined_signal / volatility
    
    return final_factor
