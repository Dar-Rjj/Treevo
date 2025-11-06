import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining trend persistence under high volatility conditions
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Identify High Volatility Periods
    # Calculate rolling volatility using high-low range and standard deviation
    high_low_range = (data['high'] - data['low']) / data['close']
    rolling_volatility = high_low_range.rolling(window=20).std()
    
    # Filter top 30% volatility days
    volatility_threshold = rolling_volatility.quantile(0.7)
    high_vol_filter = (rolling_volatility >= volatility_threshold).astype(float)
    
    # 2. Measure Price Trend Strength
    # Calculate price slope using linear regression over 10 days
    def calculate_slope(series):
        if len(series) < 10:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    price_slope = data['close'].rolling(window=10).apply(calculate_slope, raw=False)
    
    # Compute trend consistency - count consecutive same-direction days weighted by slope magnitude
    price_returns = data['close'].pct_change()
    trend_direction = np.sign(price_returns)
    
    def count_consecutive_trend(direction_series, window=10):
        result = []
        for i in range(len(direction_series)):
            if i < window:
                result.append(np.nan)
                continue
            window_data = direction_series.iloc[i-window:i]
            if window_data.isna().any():
                result.append(np.nan)
                continue
            # Count consecutive same direction at the end of window
            current_dir = window_data.iloc[-1]
            count = 0
            for j in range(len(window_data)-1, -1, -1):
                if window_data.iloc[j] == current_dir:
                    count += 1
                else:
                    break
            result.append(count)
        return pd.Series(result, index=direction_series.index)
    
    consecutive_trend = count_consecutive_trend(trend_direction)
    trend_strength = price_slope.abs() * consecutive_trend
    
    # 3. Combine Signals with Volume Confirmation
    # Multiply volatility filter with trend strength
    raw_factor = high_vol_filter * trend_strength
    
    # Apply volume confirmation
    volume_ma_20 = data['volume'].rolling(window=20).mean()
    volume_signal = (data['volume'] > volume_ma_20).astype(float)
    
    # Final factor: raw factor adjusted by volume signal
    # Use 0.5 as base when volume doesn't confirm, 1.0 when it does
    volume_adjusted_factor = raw_factor * (0.5 + 0.5 * volume_signal)
    
    return volume_adjusted_factor
