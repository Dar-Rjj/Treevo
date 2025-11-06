import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Trend Persistence factor
    Combines trend strength with volume confirmation for robust momentum signals
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Trend strength measurement
    # Price distance from 20-day linear regression
    def linear_regression_trend(series, window=20):
        x = np.arange(window)
        y = series.values
        if len(y) < window:
            return np.nan
        coeffs = np.polyfit(x, y, 1)
        predicted = coeffs[0] * (window-1) + coeffs[1]
        actual = y[-1]
        return (actual - predicted) / predicted
    
    # Calculate linear regression residuals
    data['linreg_residual'] = data['close'].rolling(window=20).apply(
        lambda x: linear_regression_trend(x), raw=False
    )
    
    # Volatility-normalized trend using 20-day ATR
    def calculate_atr(high, low, close, window=20):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    data['atr_20'] = calculate_atr(data['high'], data['low'], data['close'])
    data['price_trend'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['vol_normalized_trend'] = data['price_trend'] / data['atr_20']
    
    # Trend consistency score (days in same direction)
    def trend_consistency(returns, window=20):
        signs = np.sign(returns)
        consistency = signs.rolling(window=window).apply(
            lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0.5, raw=False
        )
        return abs(consistency - 0.5) * 2  # Scale to 0-1
    
    data['trend_consistency'] = trend_consistency(data['returns'])
    
    # Combine trend strength components
    data['trend_strength'] = (
        data['linreg_residual'].abs() * 0.4 +
        data['vol_normalized_trend'].abs() * 0.4 +
        data['trend_consistency'] * 0.2
    )
    
    # 2. Volume confirmation
    # Volume surge detection (current vs 10-day average)
    data['volume_avg_10'] = data['volume'].rolling(window=10).mean()
    data['volume_surge'] = data['volume'] / data['volume_avg_10']
    
    # Volume-trend alignment
    def volume_trend_alignment(volume, returns, window=10):
        up_days = returns > 0
        down_days = returns < 0
        
        up_volume_ratio = volume[up_days].rolling(window=window).mean() / volume.rolling(window=window).mean()
        down_volume_ratio = volume[down_days].rolling(window=window).mean() / volume.rolling(window=window).mean()
        
        # Higher volume on up days is bullish, higher volume on down days is bearish
        alignment = (up_volume_ratio - down_volume_ratio).fillna(0)
        return alignment
    
    data['volume_alignment'] = volume_trend_alignment(data['volume'], data['returns'])
    
    # Combine volume confirmation components
    data['volume_confirmation'] = (
        np.tanh(data['volume_surge'] - 1) * 0.6 +  # Normalize surge to -1 to 1
        data['volume_alignment'] * 0.4
    )
    
    # Final factor: Trend strength × Volume confirmation
    # Add direction based on price trend
    trend_direction = np.sign(data['price_trend'])
    data['factor'] = data['trend_strength'] * data['volume_confirmation'] * trend_direction
    
    # Clean and return
    factor_series = data['factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor_series
