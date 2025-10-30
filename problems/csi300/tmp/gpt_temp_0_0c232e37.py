import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Volatility-Adjusted Price-Volume Divergence Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Volatility Adjustment
    # Daily range calculation
    daily_range = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Short-term volatility (5-day)
    short_term_vol = daily_range.rolling(window=5, min_periods=3).mean()
    
    # Medium-term volatility (20-day)
    medium_term_vol = daily_range.rolling(window=20, min_periods=10).mean()
    
    # Volatility ratio
    volatility_ratio = short_term_vol / medium_term_vol
    
    # Price-Volume Divergence Components
    # Price momentum divergence
    price_return_5d = data['close'] / data['close'].shift(5) - 1
    price_return_10d = data['close'] / data['close'].shift(10) - 1
    price_momentum_divergence = price_return_5d - price_return_10d
    
    # Volume momentum divergence
    volume_ma_5d = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_ma_10d = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_ratio = volume_ma_5d / volume_ma_10d - 1
    volume_momentum = volume_ratio * np.exp(-1/5)
    
    # Trend Isolation Layer
    def calculate_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    # Short-term trend (3-day)
    short_term_trend = calculate_slope(data['close'], 3)
    
    # Medium-term trend (10-day)
    medium_term_trend = calculate_slope(data['close'], 10)
    
    # Trend-adjusted price (remove medium-term trend)
    trend_adjusted_price = data['close'] - medium_term_trend * 10
    
    # Multiplicative Interaction Engine
    # Volatility-weighted divergence
    volatility_weighted_divergence = price_momentum_divergence * volatility_ratio
    
    # Volume-enhanced signal
    volume_enhanced_signal = volatility_weighted_divergence * volume_momentum
    
    # Trend-aligned multiplier
    trend_aligned_signal = volume_enhanced_signal * short_term_trend
    
    # Dynamic Mean Reversion Trigger
    # Reversion probability score
    abs_3d_return = abs(data['close'] / data['close'].shift(3) - 1)
    reversion_probability = 1 - np.exp(-abs_3d_return)
    
    # Signal decay function (simplified - using rolling decay)
    decay_factor = pd.Series(1.0, index=data.index)
    for i in range(1, len(decay_factor)):
        decay_factor.iloc[i] = decay_factor.iloc[i-1] * np.exp(-1/10)
    
    # Final Alpha Output
    # Combine all components
    alpha_raw = trend_aligned_signal * reversion_probability * decay_factor
    
    # Normalize the alpha factor
    alpha_factor = (alpha_raw - alpha_raw.rolling(window=20, min_periods=10).mean()) / alpha_raw.rolling(window=20, min_periods=10).std()
    
    return alpha_factor
