import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Dynamic Volatility-Normalized Momentum with Volume Divergence factor
    
    Parameters:
    data: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Factor values indexed by date
    """
    
    df = data.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Volatility-Normalized Momentum
    # Short-term momentum: close_t / close_{t-5} - 1
    momentum = (close / close.shift(5)) - 1
    
    # Volatility proxy: (high_t - low_t) / close_{t-1}
    volatility = (high - low) / close.shift(1)
    
    # Normalized momentum: momentum / volatility
    # Add small epsilon to avoid division by zero
    normalized_momentum = momentum / (volatility + 1e-8)
    
    # Volume Divergence
    # Calculate volume trend: linear slope of volume over 5 days
    volume_trend = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        if i >= 4:
            window_volume = volume.iloc[i-4:i+1]
            if len(window_volume) >= 2:
                slope = linregress(range(len(window_volume)), window_volume.values)[0]
                volume_trend.iloc[i] = slope
            else:
                volume_trend.iloc[i] = 0
    
    # Price-volume direction check and divergence strength
    divergence_strength = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if pd.notna(normalized_momentum.iloc[i]) and pd.notna(volume_trend.iloc[i]):
            momentum_dir = 1 if normalized_momentum.iloc[i] > 0 else -1
            volume_dir = 1 if volume_trend.iloc[i] > 0 else -1
            
            # Check for divergence
            if momentum_dir * volume_dir < 0:  # Opposite directions
                # Divergence strength: |momentum| × |volume_trend|
                strength = abs(normalized_momentum.iloc[i]) * abs(volume_trend.iloc[i])
                divergence_strength.iloc[i] = strength
            else:
                divergence_strength.iloc[i] = 0
        else:
            divergence_strength.iloc[i] = 0
    
    # Regime Detection
    # Volatility regime: 20-day average of daily ranges
    daily_range = (high - low) / close.shift(1)
    volatility_regime = daily_range.rolling(window=20, min_periods=10).mean()
    
    # Determine regime thresholds (using median as threshold)
    regime_threshold = volatility_regime.median()
    
    # Regime-based weighting
    regime_weight = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if pd.notna(volatility_regime.iloc[i]):
            if volatility_regime.iloc[i] > regime_threshold:
                # High volatility regime: emphasize mean reversion (invert signals)
                regime_weight.iloc[i] = -1.0
            else:
                # Low volatility regime: emphasize momentum continuation
                regime_weight.iloc[i] = 1.0
        else:
            regime_weight.iloc[i] = 1.0
    
    # Alpha Combination
    # Base signal: volatility-normalized momentum
    base_signal = normalized_momentum
    
    # Volume confirmation: multiply by divergence strength when directions align
    volume_confirmation = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if pd.notna(base_signal.iloc[i]) and pd.notna(divergence_strength.iloc[i]):
            momentum_dir = 1 if base_signal.iloc[i] > 0 else -1
            volume_dir = 1 if volume_trend.iloc[i] > 0 else -1
            
            if momentum_dir * volume_dir > 0:  # Same directions
                # Use divergence strength as confirmation (scaled by 0.5 to moderate effect)
                volume_confirmation.iloc[i] = 1 + (0.5 * divergence_strength.iloc[i])
            else:
                volume_confirmation.iloc[i] = 1.0
        else:
            volume_confirmation.iloc[i] = 1.0
    
    # Final alpha: regime_weighted_signal × volume_confirmation
    regime_weighted_signal = base_signal * regime_weight
    final_alpha = regime_weighted_signal * volume_confirmation
    
    # Clean and return
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    final_alpha = final_alpha.fillna(0)
    
    return final_alpha
