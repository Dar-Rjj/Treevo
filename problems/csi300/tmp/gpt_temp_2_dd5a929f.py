import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(data):
    """
    Dynamic Volatility-Normalized Momentum with Volume Divergence factor
    """
    df = data.copy()
    
    # Volatility-Normalized Momentum Component
    # Calculate short-term momentum (5-day close-to-close returns)
    momentum = df['close'].pct_change(5)
    
    # Compute daily volatility proxy (high - low price range)
    daily_volatility = df['high'] - df['low']
    
    # Calculate 5-day average volatility (sum of daily ranges / 5)
    avg_volatility = daily_volatility.rolling(window=5).mean()
    
    # Normalize momentum by average volatility (momentum / avg_volatility)
    normalized_momentum = momentum / avg_volatility
    
    # Volume Confirmation Component
    # Calculate 5-day volume trend (linear regression slope of volume)
    def volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    volume_trend = df['volume'].rolling(window=5).apply(volume_slope, raw=False)
    
    # Determine price momentum direction (sign of 5-day returns)
    price_direction = np.sign(momentum)
    
    # Compare volume trend direction with price momentum direction
    volume_direction = np.sign(volume_trend)
    
    # Assign confirmation score (1 if same direction, -1 if opposite)
    volume_confirmation = np.where(price_direction == volume_direction, 1, -1)
    
    # Regime Detection and Weighting
    # Calculate 20-day average true range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = true_range.rolling(window=20).mean()
    
    # Classify regime: high volatility if ATR > median, low otherwise
    atr_median = atr_20.rolling(window=20).median()
    regime = np.where(atr_20 > atr_median, 'high', 'low')
    
    # Apply regime-based weighting
    regime_weight = np.where(regime == 'high', 0.3, 0.7)
    
    # Final Alpha Factor Construction
    # Combine normalized momentum with volume confirmation
    combined_factor = normalized_momentum * volume_confirmation
    
    # Multiply by regime-based weighting factor
    final_alpha = combined_factor * regime_weight
    
    return final_alpha
