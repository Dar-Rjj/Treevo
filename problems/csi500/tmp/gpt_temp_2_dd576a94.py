import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Reversal Component
    data['1_day_return'] = (data['close'].shift(1) - data['close'].shift(2)) / data['close'].shift(2)
    data['3_day_return'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    # Volatility Adjustment
    # 10-day Price Range
    data['max_high_10d'] = data['high'].rolling(window=10, min_periods=1).max()
    data['min_low_10d'] = data['low'].rolling(window=10, min_periods=1).min()
    data['price_range_10d'] = (data['max_high_10d'] - data['min_low_10d']) / data['close']
    
    # 10-day Average True Range
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_10d'] = data['true_range'].rolling(window=10, min_periods=1).mean()
    
    # Liquidity Confirmation
    # Dollar Volume Slope (5-day linear regression slope)
    data['dollar_volume'] = data['close'] * data['volume']
    
    def linear_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    data['dollar_volume_slope'] = data['dollar_volume'].rolling(window=5, min_periods=2).apply(linear_slope, raw=False)
    
    # Price Efficiency (5-day average of (High-Low)/Close)
    data['price_efficiency'] = ((data['high'] - data['low']) / data['close']).rolling(window=5, min_periods=1).mean()
    
    # Signal Combination
    data['reversal1'] = data['1_day_return'] / data['price_range_10d']
    data['reversal2'] = data['3_day_return'] / data['atr_10d']
    data['liquidity1'] = data['dollar_volume_slope']
    data['liquidity2'] = 1 / data['price_efficiency']
    
    # Final Alpha
    alpha = data['reversal1'] + data['reversal2'] + data['liquidity1'] + data['liquidity2']
    
    return alpha
