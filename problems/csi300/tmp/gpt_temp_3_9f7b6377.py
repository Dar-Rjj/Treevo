import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Calculate Volume-Weighted Price Returns
    data['next_day_open'] = data['open'].shift(-1)
    data['simple_returns'] = (data['next_day_open'] - data['close']) / data['close']
    data['volume_weighted_returns'] = data['simple_returns'] * data['volume']

    # Identify Volume Surge Days
    data['daily_volume_change'] = data['volume'].diff()
    data['rolling_volume_mean'] = data['volume'].rolling(window=5).mean()
    data['volume_surge'] = (data['volume'] > data['rolling_volume_mean']).astype(int)

    # Calculate Adaptive Volatility
    data['daily_returns'] = data['close'].pct_change()
    def dynamic_lookback_volatility(x):
        recent_volatility = x.rolling(window=30).std().iloc[-1]
        if recent_volatility > 0.01:
            lookback_period = 20
        elif recent_volatility > 0.005:
            lookback_period = 40
        else:
            lookback_period = 60
        return x.rolling(window=lookback_period).std().dropna()
    
    data['adaptive_volatility'] = data['daily_returns'].rolling(window=30).apply(dynamic_lookback_volatility, raw=False)
    
    data['volume_moving_average'] = data['volume'].rolling(window=20).mean()
    data['volume_z_score'] = (data['volume'] - data['volume_moving_average']) / data['volume'].rolling(window=20).std()
    data['adjusted_volatility'] = data['adaptive_volatility'] * (1 + abs(data['volume_z_score']))

    # Refine Surge Factors
    data['volume_surge_ratio'] = data['volume'] / data['volume'].shift(1)
    def refine_surge_factor(ratio):
        if ratio > 2.5:
            return 1.8
        elif ratio > 2.0:
            return 1.5
        elif ratio > 1.5:
            return 1.2
        else:
            return 1.0
    data['refined_surge_factor'] = data['volume_surge_ratio'].apply(refine_surge_factor)

    # Adjust Volume-Weighted Returns by Adaptive Volatility
    data['adjusted_returns'] = data['volume_weighted_returns'] / data['adjusted_volatility']

    # Combine Adjusted Returns with Refined Volume Surge Indicator
    data['factor_value'] = data['adjusted_returns'] * np.where(data['volume_surge'] == 1, data['refined_surge_factor'], 1.0)

    return data['factor_value'].dropna()
