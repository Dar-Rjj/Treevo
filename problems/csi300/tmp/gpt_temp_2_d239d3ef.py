import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns based on Close/Open
    daily_returns = data['close'] / data['open'] - 1
    
    # Short-term Acceleration (3-day)
    short_term_accel = (daily_returns.rolling(window=3).sum() - 
                       daily_returns.shift(1).rolling(window=3).sum())
    
    # Medium-term Acceleration (5-day)
    medium_term_accel = (daily_returns.rolling(window=5).sum() - 
                        daily_returns.shift(1).rolling(window=5).sum())
    
    # Acceleration Divergence
    accel_divergence = short_term_accel - medium_term_accel
    
    # Volume Trend Ratio
    volume_trend_ratio = (data['volume'].rolling(window=3).mean() / 
                         data['volume'].rolling(window=5).mean())
    
    # Price-Volume Alignment (3-day correlation)
    price_changes = data['close'] / data['close'].shift(1) - 1
    
    def rolling_corr(x, y, window):
        corrs = []
        for i in range(len(x)):
            if i < window - 1:
                corrs.append(np.nan)
            else:
                start_idx = i - window + 1
                end_idx = i + 1
                x_window = x.iloc[start_idx:end_idx]
                y_window = y.iloc[start_idx:end_idx]
                if len(x_window) == window and len(y_window) == window:
                    corr = np.corrcoef(x_window, y_window)[0, 1]
                    corrs.append(corr if not np.isnan(corr) else 0)
                else:
                    corrs.append(np.nan)
        return pd.Series(corrs, index=x.index)
    
    price_volume_alignment = rolling_corr(price_changes, data['volume'], 3)
    
    # Volume Efficiency
    volume_efficiency = (volume_trend_ratio - 1) * price_volume_alignment
    
    # Core Signal
    core_signal = accel_divergence * volume_efficiency
    
    # Volatility Adjustment (3-day std of daily range)
    daily_range = data['high'] - data['low']
    volatility = daily_range.rolling(window=3).std()
    
    # Apply volatility adjustment
    volatility_adjusted = core_signal / volatility.replace(0, np.nan)
    
    # Linear Regression Slope (3-day)
    def rolling_slope(series, window):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                start_idx = i - window + 1
                end_idx = i + 1
                y_window = series.iloc[start_idx:end_idx]
                if len(y_window) == window:
                    x = np.arange(window)
                    slope = linregress(x, y_window).slope
                    slopes.append(slope if not np.isnan(slope) else 0)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    slope_3day = rolling_slope(data['close'], 3)
    
    # Final Alpha
    final_alpha = volatility_adjusted * slope_3day
    
    return final_alpha
