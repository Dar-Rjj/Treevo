import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def heuristics_v2(data):
    # Calculate EMAs
    short_window = 5
    long_window = 20
    data['short_ema'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['long_ema'] = data['close'].ewm(span=long_window, adjust=False).mean()

    # Volume-Weighted Close Price
    data['vw_close'] = (data['close'] * data['volume']) / data['volume'].sum()

    # Volume-Weighted High-Low Difference
    data['hl_diff_vol_weighted'] = (data['high'] - data['low']) * data['volume']
    data['hl_diff_vol_weighted_sum'] = data['hl_diff_vol_weighted'].rolling(window=long_window).sum()

    # Time-Series Momentum
    momentum_window = 10
    data['momentum'] = (data['close'] - data['close'].shift(momentum_window)) / data['close'].shift(momentum_window)

    # Dynamic Factor Weighting
    def dynamic_weighting(momentum, sector_performance):
        if momentum > 0:
            weight_short = 0.7
            weight_long = 0.3
        else:
            weight_short = 0.3
            weight_long = 0.7
        
        if sector_performance > 0:
            weight_short += 0.1
            weight_long -= 0.1
        else:
            weight_short -= 0.1
            weight_long += 0.1
        
        return weight_short, weight_long

    # Assume we have sector performance data
    sector_performance = (data['close'] - data['close'].shift(long_window)) / data['close'].shift(long_window)
    data['weight_short'], data['weight_long'] = zip(*data.apply(lambda row: dynamic_weighting(row['momentum'], sector_performance), axis=1))

    # Exponential Smoothing
    alpha = 0.2
    data['smoothed_short_ema'] = data['short_ema'].ewm(alpha=alpha, adjust=False).mean()
    data['smoothed_long_ema'] = data['long_ema'].ewm(alpha=alpha, adjust=False).mean()

    # Combine factors
    data['factor_value'] = (data['weight_short'] * data['smoothed_short_ema'] + 
                            data['weight_long'] * data['smoothed_long_ema'] +
                            data['momentum'] + 
                            sector_performance)

    return data['factor_value']

# Example usage
# data = pd.DataFrame({
#     'open': [100, 102, 98, 101, 103],
#     'high': [105, 107, 104, 106, 108],
#     'low': [95, 97, 96, 98, 100],
#     'close': [103, 105, 102, 104, 106],
#     'amount': [10000, 12000, 9000, 11000, 13000],
#     'volume': [1000, 1200, 900, 1100, 1300]
# })
# factor_values = heuristics_v2(data)
# print(factor_values)
