import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-adjusted volume trend with momentum divergence patterns.
    """
    # Volatility-Adjusted Volume Trend
    # Calculate Rolling Volume Trend
    vol_ma_20 = df['volume'].rolling(window=20, min_periods=10).mean()
    vol_ma_5 = df['volume'].rolling(window=5, min_periods=3).mean()
    volume_trend = vol_ma_5 / vol_ma_20
    
    # Calculate Price Volatility
    daily_range = (df['high'] - df['low']) / df['close']
    price_volatility = daily_range.rolling(window=10, min_periods=5).std()
    
    # Calculate Recent Return Sign
    recent_return = (df['close'] / df['close'].shift(3)) - 1
    return_sign = np.sign(recent_return)
    
    # Volatility-Adjusted Volume Trend Component
    volatility_adjusted_volume = volume_trend * price_volatility * return_sign
    
    # Momentum Divergence with Volume Confirmation
    # Calculate Price Momentum
    short_momentum = (df['close'] / df['close'].shift(5)) - 1
    medium_momentum = (df['close'] / df['close'].shift(20)) - 1
    
    # Calculate Volume Momentum
    volume_change = df['volume'] / vol_ma_5
    
    # Volume Trend Slope (10-day linear regression)
    def volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    volume_trend_slope = df['volume'].rolling(window=10, min_periods=5).apply(
        volume_slope, raw=False
    )
    
    # Identify Divergence Patterns
    bullish_divergence = (
        (short_momentum < 0) & 
        (volume_change > 1) & 
        (volume_trend_slope > 0)
    ).astype(float)
    
    bearish_divergence = (
        (short_momentum > 0) & 
        (volume_change < 1) & 
        (volume_trend_slope < 0)
    ).astype(float)
    
    momentum_divergence = bullish_divergence - bearish_divergence
    
    # Combine both components
    alpha_factor = volatility_adjusted_volume * 0.6 + momentum_divergence * 0.4
    
    # Clean and return
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    return alpha_factor
