import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Identify Recent Price Extremes
    data['highest_high_10d'] = data['high'].rolling(window=10, min_periods=5).max()
    data['lowest_low_10d'] = data['low'].rolling(window=10, min_periods=5).min()
    
    # Calculate Price Position Ratio
    data['dist_to_low'] = data['close'] - data['lowest_low_10d']
    data['dist_to_high'] = data['highest_high_10d'] - data['close']
    
    # Avoid division by zero
    data['dist_to_high'] = data['dist_to_high'].replace(0, np.nan)
    data['price_position_ratio'] = data['dist_to_low'] / data['dist_to_high']
    
    # Incorporate Volatility Adjustment
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['avg_true_range_5d'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    
    # Avoid division by zero in ATR
    data['avg_true_range_5d'] = data['avg_true_range_5d'].replace(0, np.nan)
    data['volatility_adjusted_ratio'] = data['price_position_ratio'] / data['avg_true_range_5d']
    
    # Add Volume-Weighted Momentum Component
    data['roc_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Calculate Volume Trend Slope using linear regression
    def volume_slope(volume_series):
        if len(volume_series) < 3:
            return np.nan
        x = np.arange(len(volume_series))
        slope = np.polyfit(x, volume_series, 1)[0]
        return slope / np.mean(volume_series)  # Normalize by average volume
    
    data['volume_trend_slope'] = data['volume'].rolling(window=5, min_periods=3).apply(
        volume_slope, raw=False
    )
    data['volume_weighted_momentum'] = data['roc_5d'] * data['volume_trend_slope']
    
    # Combine Components with Liquidity Filter
    data['turnover_rate'] = data['volume'] / data['amount']
    data['avg_turnover_10d'] = data['turnover_rate'].rolling(window=10, min_periods=5).mean()
    
    # Avoid division by zero in turnover
    data['avg_turnover_10d'] = data['avg_turnover_10d'].replace(0, np.nan)
    data['turnover_ratio'] = data['turnover_rate'] / data['avg_turnover_10d']
    
    # Final factor calculation
    data['factor'] = (data['volatility_adjusted_ratio'] + data['volume_weighted_momentum']) * data['turnover_ratio']
    
    return data['factor']
