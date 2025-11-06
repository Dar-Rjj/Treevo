import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Dynamic Volatility-Normalized Momentum with Volume Divergence alpha factor
    
    Parameters:
    data: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    pandas Series with alpha factor values
    """
    
    # Extract price and volume data
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # Volatility-Normalized Momentum Calculation
    # Short-term price momentum (5-day return)
    momentum_5d = close / close.shift(5) - 1
    
    # Volatility normalization using daily range
    daily_range = high - low
    volatility_5d = daily_range.rolling(window=5).std()
    
    # Avoid division by zero
    volatility_normalized_momentum = momentum_5d / (volatility_5d + 1e-8)
    
    # Volume Divergence Detection
    # Volume trend relative to 5-day moving average
    volume_ma_5d = volume.rolling(window=5).mean()
    volume_trend = volume / (volume_ma_5d + 1e-8)
    
    # Detect divergence: when momentum and volume trend have opposite signs
    volume_divergence = np.where(
        np.sign(momentum_5d) != np.sign(volume_trend - 1), 
        np.abs(volume_trend - 1), 
        0
    )
    
    # Regime-Based Weighting
    # Detect volatility regime using rolling quantiles of daily range
    daily_range_quantile = daily_range.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], 
        raw=False
    )
    
    # Classify regimes: high volatility when above 70th percentile
    high_vol_regime = (daily_range_quantile > 0.7).astype(float)
    low_vol_regime = (daily_range_quantile < 0.3).astype(float)
    
    # Apply adaptive weights
    # Higher weight to volume divergence in high volatility
    # Higher weight to momentum in low volatility
    volume_div_weight = 0.7 * high_vol_regime + 0.3 * low_vol_regime
    momentum_weight = 0.3 * high_vol_regime + 0.7 * low_vol_regime
    
    # Signal Combination
    # Combine momentum and volume divergence with regime-based weights
    combined_signal = (
        momentum_weight * volatility_normalized_momentum +
        volume_div_weight * volume_divergence * np.sign(volatility_normalized_momentum)
    )
    
    # Final alpha factor
    alpha_factor = combined_signal
    
    return alpha_factor
