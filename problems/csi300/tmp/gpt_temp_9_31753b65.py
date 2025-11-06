import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-adjusted momentum, intraday trend persistence,
    volume-price divergence, and acceleration-deceleration signals.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility-Adjusted Price Momentum
    # Calculate momentum using 5, 10, 20-day periods
    momentum_5 = data['close'].pct_change(5)
    momentum_10 = data['close'].pct_change(10)
    momentum_20 = data['close'].pct_change(20)
    
    # Calculate historical volatility using high-low range (20-day rolling std)
    daily_range = (data['high'] - data['low']) / data['close']
    volatility = daily_range.rolling(window=20, min_periods=10).std()
    
    # Volatility-adjusted momentum (average of different periods)
    vol_adj_momentum = ((momentum_5 + momentum_10 + momentum_20) / 3) / (volatility + 1e-8)
    
    # 2. Intraday Trend Persistence
    # Calculate intraday strength
    intraday_strength = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Calculate rolling mean and autocorrelation
    intraday_rolling_mean = intraday_strength.rolling(window=5, min_periods=3).mean()
    
    # Calculate autocorrelation using rolling correlation with lag
    intraday_lagged = intraday_strength.shift(1)
    intraday_autocorr = intraday_strength.rolling(window=10, min_periods=5).corr(intraday_lagged)
    
    # Persistence measure weighted by recent magnitude
    intraday_persistence = intraday_autocorr * intraday_rolling_mean.abs()
    
    # 3. Volume-Price Divergence Factor
    def calculate_trend(series, window=10):
        """Calculate linear regression slope for trend"""
        def linreg_slope(x):
            if len(x) < 2:
                return np.nan
            return stats.linregress(np.arange(len(x)), x)[0]
        return series.rolling(window=window, min_periods=5).apply(linreg_slope, raw=False)
    
    price_trend = calculate_trend(data['close'], window=10)
    volume_trend = calculate_trend(data['volume'], window=10)
    
    # Volume-price divergence (when price and volume trends diverge)
    volume_price_divergence = price_trend / (volume_trend + 1e-8)
    
    # 4. Acceleration-Deceleration Indicator
    # First derivative (velocity)
    price_velocity = data['close'].diff()
    
    # Second derivative (acceleration)
    price_acceleration = price_velocity.diff()
    
    # Combine acceleration with momentum direction
    acceleration_signal = price_acceleration * np.sign(momentum_5)
    
    # 5. Combine all factors with weights
    # Normalize each component using z-score
    def normalize_series(series):
        return (series - series.rolling(window=50, min_periods=20).mean()) / (series.rolling(window=50, min_periods=20).std() + 1e-8)
    
    # Apply normalization
    vol_adj_momentum_norm = normalize_series(vol_adj_momentum)
    intraday_persistence_norm = normalize_series(intraday_persistence)
    volume_price_divergence_norm = normalize_series(volume_price_divergence)
    acceleration_signal_norm = normalize_series(acceleration_signal)
    
    # Final factor combination with weights
    final_factor = (
        0.4 * vol_adj_momentum_norm +
        0.3 * intraday_persistence_norm +
        0.2 * volume_price_divergence_norm +
        0.1 * acceleration_signal_norm
    )
    
    return final_factor
