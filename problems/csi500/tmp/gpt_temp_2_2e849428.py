import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate 1-day lagged return
    data['lagged_return_1d'] = (data['close'].shift(1) - data['close'].shift(2)) / data['close'].shift(2)
    
    # Calculate 3-day rolling return
    data['rolling_return_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    # Calculate 10-day price range
    data['high_10d'] = data['high'].rolling(window=10, min_periods=10).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=10).min()
    data['price_range_10d'] = (data['high_10d'] - data['low_10d']) / data['close']
    
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate 10-day Average True Range
    data['atr_10d'] = data['true_range'].rolling(window=10, min_periods=10).mean()
    
    # Calculate dollar volume
    data['dollar_volume'] = data['close'] * data['volume']
    
    # Calculate 5-day dollar volume slope using linear regression
    def linear_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        return slope
    
    data['dollar_volume_slope_5d'] = data['dollar_volume'].rolling(window=5, min_periods=5).apply(linear_slope, raw=False)
    
    # Calculate 5-day price efficiency
    data['daily_range_ratio'] = (data['high'] - data['low']) / data['close']
    data['price_efficiency_5d'] = data['daily_range_ratio'].rolling(window=5, min_periods=5).mean()
    
    # Calculate components
    data['reversal1'] = data['lagged_return_1d'] / data['price_range_10d']
    data['reversal2'] = data['rolling_return_3d'] / data['atr_10d']
    data['liquidity1'] = data['dollar_volume_slope_5d']
    data['liquidity2'] = 1 / data['price_efficiency_5d']
    
    # Calculate final alpha
    data['alpha'] = data['reversal1'] + data['reversal2'] + data['liquidity1'] + data['liquidity2']
    
    return data['alpha']
