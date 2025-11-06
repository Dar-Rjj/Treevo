import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Volatility-Normalized Momentum with Volume Divergence alpha factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Volatility-Normalized Momentum Calculation
    # Compute 5-day close returns (momentum)
    momentum_5d = data['close'].pct_change(5)
    
    # Calculate daily range (high-low) and 5-day average range as volatility
    daily_range = data['high'] - data['low']
    volatility_5d = daily_range.rolling(window=5).mean()
    
    # Normalize momentum by volatility
    normalized_momentum = momentum_5d / (volatility_5d + 1e-8)
    
    # Volume Divergence Detection
    # Calculate 5-day volume trend using linear regression slope
    def volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(len(volume_series))
        return np.polyfit(x, volume_series, 1)[0]
    
    volume_trend = data['volume'].rolling(window=5).apply(volume_slope, raw=False)
    
    # Compare volume trend direction with price momentum direction
    volume_direction = np.sign(volume_trend)
    price_direction = np.sign(momentum_5d)
    
    # Identify divergence when directions oppose
    volume_divergence = volume_direction * price_direction
    # Negative values indicate divergence (opposing directions)
    
    # Regime-Based Weighting
    # Calculate 20-day ATR for volatility regime detection
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20d = true_range.rolling(window=20).mean()
    
    # Normalize ATR by price to get relative volatility
    relative_volatility = atr_20d / data['close']
    
    # Define volatility regimes (high vs low volatility)
    volatility_regime = relative_volatility.rolling(window=20).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else (-1 if x.iloc[-1] < x.quantile(0.3) else 0), 
        raw=False
    )
    
    # Signal Combination
    # Strong signal: normalized momentum with volume confirmation (no divergence)
    strong_signal = normalized_momentum.copy()
    # Reduce signal strength when volume divergence exists
    strong_signal = strong_signal * np.where(volume_divergence < 0, 0.3, 1.0)
    
    # Weak signal: normalized momentum alone
    weak_signal = normalized_momentum.copy()
    
    # Apply regime weights to final alpha factor
    # High volatility: emphasize mean reversion (inverse signal)
    # Low volatility: emphasize momentum continuation (direct signal)
    regime_weight = np.where(volatility_regime == 1, -0.7,  # High vol: mean reversion
                   np.where(volatility_regime == -1, 1.0,   # Low vol: momentum
                           0.0))                           # Normal: neutral
    
    # Combine signals with regime weighting
    alpha_factor = strong_signal * (1 + regime_weight) + weak_signal * regime_weight
    
    # Normalize the final factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20).mean()) / (alpha_factor.rolling(window=20).std() + 1e-8)
    
    return alpha_factor
