import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Range Efficiency
    data['close_change'] = abs(data['close'] - data['prev_close'])
    data['range_efficiency'] = data['close_change'] / data['true_range']
    
    # Range Efficiency Acceleration
    data['re_diff1'] = data['range_efficiency'].diff()
    data['re_diff2'] = data['re_diff1'].diff()
    data['range_efficiency_acceleration'] = data['re_diff2']
    
    # Volume-Weighted Momentum
    data['short_term_return'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_return'] = data['close'] / data['close'].shift(8) - 1
    data['volume_ma'] = data['volume'].rolling(window=20, min_periods=1).mean().shift(1)
    data['volume_flag'] = (data['volume'] > data['volume_ma']).astype(int)
    data['weighted_return'] = 0.7 * data['short_term_return'] + 0.3 * data['medium_term_return'] * data['volume_flag']
    
    # Acceleration-Volume Divergence
    data['price_change'] = data['close'] - data['prev_close']
    data['price_volume_ratio'] = data['price_change'] / data['volume']
    data['price_acceleration'] = data['price_change'].diff().diff()
    
    # Calculate 5-day rolling correlation
    correlation_window = 5
    correlations = []
    for i in range(len(data)):
        if i < correlation_window:
            correlations.append(0)
        else:
            window_data = data.iloc[i-correlation_window+1:i+1]
            corr = window_data['range_efficiency_acceleration'].corr(window_data['price_volume_ratio'])
            correlations.append(corr if not np.isnan(corr) else 0)
    
    data['acceleration_divergence_corr'] = correlations
    
    # Composite Alpha
    data['base'] = data['range_efficiency_acceleration'] * data['weighted_return']
    data['divergence_multiplier'] = 1 + np.sign(data['acceleration_divergence_corr']) * abs(data['acceleration_divergence_corr'])
    data['alpha'] = data['base'] * data['divergence_multiplier']
    
    # Return the alpha series with same index as input
    return data['alpha']
