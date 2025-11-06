import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Price Movement Asymmetry
    # Upward Intensity: Average(Close_t - Close_t-1 | Close_t > Close_t-1) over 5 days
    upward_moves = data['close'].diff()
    upward_intensity = upward_moves.where(upward_moves > 0).rolling(window=5, min_periods=1).mean()
    
    # Downward Intensity: Average(Close_t-1 - Close_t | Close_t < Close_t-1) over 5 days
    downward_moves = -upward_moves
    downward_intensity = downward_moves.where(downward_moves > 0).rolling(window=5, min_periods=1).mean()
    
    # Volume-Price Divergence
    # High Volume Rejection: (High_t-1 - Close_t-1) / (High_t-1 - Low_t-1) * Volume_t-1 / 10-day volume median
    price_range = data['high'] - data['low']
    rejection_ratio = (data['high'].shift(1) - data['close'].shift(1)) / price_range.shift(1)
    volume_median_10d = data['volume'].rolling(window=10, min_periods=1).median()
    high_volume_rejection = rejection_ratio * data['volume'].shift(1) / volume_median_10d.shift(1)
    
    # Low Volume Breakout: |Close_t-1 - Open_t-1| / (High_t-1 - Low_t-1) * 3-day volume average / 20-day volume average
    open_close_diff = abs(data['close'].shift(1) - data['open'].shift(1))
    volume_avg_3d = data['volume'].rolling(window=3, min_periods=1).mean()
    volume_avg_20d = data['volume'].rolling(window=20, min_periods=1).mean()
    low_volume_breakout = (open_close_diff / price_range.shift(1)) * (volume_avg_3d.shift(1) / volume_avg_20d.shift(1))
    
    # Market Regime
    # Volatility State: (High_t-1 - Low_t-1) / 10-day average range
    avg_range_10d = price_range.rolling(window=10, min_periods=1).mean()
    volatility_state = price_range.shift(1) / avg_range_10d.shift(1)
    
    # Trend Quality: Sign correlation between daily returns over 8 days
    def sign_correlation(returns_series):
        if len(returns_series) < 2:
            return np.nan
        signs = np.sign(returns_series)
        return np.corrcoef(signs[:-1], signs[1:])[0, 1] if len(signs) > 1 else np.nan
    
    trend_quality = data['returns'].rolling(window=8, min_periods=2).apply(
        sign_correlation, raw=False
    )
    
    # Combine components into final factor
    # Normalize components to similar scales
    upward_intensity_norm = upward_intensity / upward_intensity.rolling(window=20, min_periods=1).std()
    downward_intensity_norm = downward_intensity / downward_intensity.rolling(window=20, min_periods=1).std()
    high_volume_rejection_norm = high_volume_rejection / high_volume_rejection.rolling(window=20, min_periods=1).std()
    low_volume_breakout_norm = low_volume_breakout / low_volume_breakout.rolling(window=20, min_periods=1).std()
    volatility_state_norm = volatility_state / volatility_state.rolling(window=20, min_periods=1).std()
    trend_quality_norm = trend_quality / trend_quality.rolling(window=20, min_periods=1).std()
    
    # Final factor: Price asymmetry + Volume divergence adjusted by market regime
    factor = (
        (upward_intensity_norm - downward_intensity_norm) * 
        (1 + high_volume_rejection_norm - low_volume_breakout_norm) *
        (1 + volatility_state_norm) *
        (1 + trend_quality_norm)
    )
    
    return factor
