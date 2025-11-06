import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum-volume alignment with adaptive volatility scaling and regime awareness.
    
    Factor interpretation:
    - Multi-timeframe momentum fusion: Combines 3-day, 6-day, and 12-day momentum with exponential weighting
      to emphasize recent trends while maintaining multi-horizon perspective
    - Volume trend persistence: Uses 5-day vs 15-day volume trend slopes to capture acceleration/deceleration
      in trading activity rather than simple averages
    - Volatility-scaled range efficiency: Measures price efficiency within daily ranges, scaled by 
      20-day rolling volatility to adapt to current market conditions
    - Regime-aware thresholds: Applies dynamic percentile-based thresholds to identify significant moves
      and filter noise across different market environments
    - Multiplicative alignment targets stocks with consistent momentum across timeframes, 
      accelerating volume participation, and efficient price discovery in current volatility regime
    """
    
    # Multi-timeframe momentum fusion with exponential decay
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_6d = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)
    momentum_12d = (df['close'] - df['close'].shift(12)) / df['close'].shift(12)
    
    # Exponential weighting: recent momentum gets higher weight (3d:0.5, 6d:0.3, 12d:0.2)
    momentum_fusion = (0.5 * momentum_3d + 0.3 * momentum_6d + 0.2 * momentum_12d)
    
    # Volume trend persistence using rolling regression slopes
    def volume_trend_slope(series, window):
        x = np.arange(window)
        return series.rolling(window=window).apply(
            lambda y: np.polyfit(x, y, 1)[0] if len(y) == window else np.nan, 
            raw=True
        )
    
    volume_trend_5d = volume_trend_slope(df['volume'], 5)
    volume_trend_15d = volume_trend_slope(df['volume'], 15)
    
    # Volume acceleration: short-term trend relative to medium-term trend
    volume_acceleration = volume_trend_5d / (abs(volume_trend_15d) + 1e-7)
    
    # Volatility-scaled range efficiency
    daily_range_efficiency = abs(df['close'] - df['close'].shift(1)) / ((df['high'] - df['low']) + 1e-7)
    efficiency_7d = daily_range_efficiency.rolling(window=7).mean()
    
    # Rolling volatility using close-to-close returns
    returns = df['close'].pct_change()
    volatility_20d = returns.rolling(window=20).std()
    
    # Volatility-scaled efficiency: efficient movement relative to current volatility regime
    volatility_scaled_efficiency = efficiency_7d / (volatility_20d + 1e-7)
    
    # Regime-aware thresholding using rolling percentiles
    momentum_threshold = momentum_fusion.rolling(window=30).apply(
        lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) > 0 else 0, 
        raw=False
    )
    
    volume_threshold = volume_acceleration.rolling(window=30).apply(
        lambda x: np.percentile(x.dropna(), 60) if len(x.dropna()) > 0 else 0, 
        raw=False
    )
    
    # Apply adaptive thresholds
    momentum_signal = np.where(momentum_fusion > momentum_threshold, momentum_fusion, 0)
    volume_signal = np.where(volume_acceleration > volume_threshold, volume_acceleration, 0)
    
    # Combined factor with multiplicative alignment
    alpha_factor = momentum_signal * volume_signal * volatility_scaled_efficiency
    
    return alpha_factor
