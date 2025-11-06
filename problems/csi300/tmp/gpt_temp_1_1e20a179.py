import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Volatility-Normalized Momentum with Volume Divergence factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Volatility-Normalized Momentum Calculation
    # Compute short-term momentum (5-day close returns)
    momentum_5d = data['close'].pct_change(5)
    
    # Calculate volatility using daily range (high-low)
    daily_range = (data['high'] - data['low']) / data['close'].shift(1)
    range_volatility = daily_range.rolling(window=5, min_periods=3).std()
    
    # Normalize momentum by volatility (momentum / range)
    normalized_momentum = momentum_5d / (range_volatility + 1e-8)
    
    # Volume Divergence Detection
    # Calculate volume trend (5-day volume slope)
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    volume_slope = data['volume'].rolling(window=5, min_periods=3).apply(
        calc_slope, raw=False
    )
    
    # Compare volume trend direction with price momentum
    price_momentum_direction = np.sign(momentum_5d)
    volume_direction = np.sign(volume_slope)
    
    # Identify divergence when directions oppose
    volume_divergence = (price_momentum_direction != volume_direction).astype(int)
    
    # Regime-Based Weighting
    # Detect volatility regime using 20-day ATR
    atr = (
        (data['high'] - data['low']).rolling(window=20).mean() / 
        data['close'].rolling(window=20).mean()
    )
    
    # High volatility: emphasize mean reversion signals
    # Low volatility: emphasize momentum continuation
    volatility_regime = atr.rolling(window=20, min_periods=10).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else -1 if x.iloc[-1] < x.quantile(0.3) else 0
    )
    
    # Signal Combination
    # Strong signal: normalized momentum + volume confirmation
    strong_signal = normalized_momentum * (1 + volume_divergence)
    
    # Weak signal: normalized momentum alone
    weak_signal = normalized_momentum
    
    # Apply regime weights to final alpha factor
    def apply_regime_weights(strong_sig, weak_sig, regime):
        if regime == 1:  # High volatility - mean reversion
            return -strong_sig * 1.5
        elif regime == -1:  # Low volatility - momentum continuation
            return strong_sig * 1.2
        else:  # Normal regime
            return (strong_sig + weak_sig) / 2
    
    alpha_factor = pd.Series(
        [apply_regime_weights(strong_signal.iloc[i], weak_signal.iloc[i], volatility_regime.iloc[i]) 
         for i in range(len(data))],
        index=data.index
    )
    
    # Clean and normalize the final factor
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20, min_periods=10).mean()) / \
                   (alpha_factor.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return alpha_factor
