import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-adjusted momentum, volume-weighted reversal,
    amplitude-adjusted breakout, and volume-price divergence signals.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Volatility-Adjusted Price Momentum
    # Calculate price momentum
    short_term_return = close / close.shift(5) - 1
    medium_term_return = close / close.shift(20) - 1
    
    # Calculate volatility adjustment
    daily_range = high / low - 1
    rolling_volatility = daily_range.rolling(window=20).std()
    
    # Risk-adjusted momentum
    risk_adj_short_momentum = short_term_return / rolling_volatility
    risk_adj_medium_momentum = medium_term_return / rolling_volatility
    
    # Volume-Weighted Price Reversal
    # Calculate recent price changes
    daily_return = close / close.shift(1) - 1
    three_day_return = close / close.shift(3) - 1
    
    # Calculate volume profile
    volume_ratio = volume / volume.shift(1)
    volume_trend = volume / volume.rolling(window=5).mean().shift(1)
    
    # Volume-weighted reversal signals
    vol_weighted_daily_reversal = daily_return * volume_ratio
    vol_weighted_trend_reversal = three_day_return * volume_trend
    
    # Amplitude-Adjusted Breakout Factor
    # Identify price levels
    recent_high = high.rolling(window=10).max()
    recent_low = low.rolling(window=10).min()
    
    # Calculate breakout strength
    upward_breakout = close / recent_high - 1
    downward_breakout = close / recent_low - 1
    
    # Adjust for market amplitude
    avg_daily_range = daily_range.rolling(window=20).mean()
    amplitude_adj_breakout = upward_breakout / avg_daily_range
    
    # Volume-Price Divergence Factor
    def linear_regression_slope(series, window):
        """Calculate linear regression slope over rolling window"""
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    # Calculate trends and accelerations
    price_slope = linear_regression_slope(close, 10)
    volume_slope = linear_regression_slope(volume, 10)
    
    price_acceleration = price_slope.diff()
    volume_acceleration = volume_slope.diff()
    
    # Detect divergence patterns
    trend_alignment = price_slope * volume_slope
    acceleration_divergence = price_acceleration - volume_acceleration
    
    # Combine all factors with equal weighting
    factor = (
        risk_adj_short_momentum.fillna(0) +
        risk_adj_medium_momentum.fillna(0) +
        vol_weighted_daily_reversal.fillna(0) +
        vol_weighted_trend_reversal.fillna(0) +
        amplitude_adj_breakout.fillna(0) +
        trend_alignment.fillna(0) +
        acceleration_divergence.fillna(0)
    ) / 7
    
    return factor
