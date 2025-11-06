import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Volatility-Normalized Momentum with Volume Divergence alpha factor
    """
    # Volatility-Normalized Momentum Calculation
    # Compute short-term momentum (5-day close returns)
    momentum = df['close'].pct_change(periods=5)
    
    # Calculate volatility using daily range (high-low)
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    volatility = daily_range.rolling(window=5, min_periods=3).mean()
    
    # Normalize momentum by volatility (momentum / range)
    normalized_momentum = momentum / (volatility + 1e-8)
    
    # Volume Divergence Detection
    # Calculate volume trend (5-day volume slope)
    volume_trend = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    
    # Compare volume trend direction with price momentum
    price_trend = df['close'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    
    # Identify divergence when directions oppose
    volume_divergence = np.sign(volume_trend) != np.sign(price_trend)
    
    # Regime-Based Weighting
    # Detect volatility regime using 20-day ATR
    atr = (df['high'] - df['low']).rolling(window=20, min_periods=10).mean() / df['close'].shift(1)
    volatility_regime = atr.rolling(window=20, min_periods=10).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else (-1 if x.iloc[-1] < x.quantile(0.3) else 0), 
        raw=False
    )
    
    # Signal Combination
    # Strong signal: normalized momentum + volume confirmation
    strong_signal = normalized_momentum * volume_divergence.astype(float)
    
    # Weak signal: normalized momentum alone
    weak_signal = normalized_momentum
    
    # Apply regime weights to final alpha factor
    # High volatility: emphasize mean reversion (negative weight on momentum)
    # Low volatility: emphasize momentum continuation (positive weight on momentum)
    regime_weight = np.where(volatility_regime == 1, -0.7, 
                           np.where(volatility_regime == -1, 1.0, 0.3))
    
    # Combine signals with regime weighting
    alpha_factor = (strong_signal * 0.6 + weak_signal * 0.4) * regime_weight
    
    return alpha_factor
