import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Price Fractal Efficiency factor
    Combines price movement efficiency with volume participation and trend alignment
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range Component
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate Net Price Change (10-day)
    data['net_price_change'] = data['close'] - data['close'].shift(10)
    
    # Calculate Volume Distribution Skew
    volume_window = 21
    data['volume_25p'] = data['volume'].rolling(window=volume_window, min_periods=10).quantile(0.25)
    data['volume_75p'] = data['volume'].rolling(window=volume_window, min_periods=10).quantile(0.75)
    data['volume_skew'] = np.where(
        data['volume'] > data['volume_75p'],
        (data['volume'] - data['volume_75p']) / (data['volume_75p'] - data['volume_25p'] + 1e-8),
        np.where(
            data['volume'] < data['volume_25p'],
            (data['volume'] - data['volume_25p']) / (data['volume_75p'] - data['volume_25p'] + 1e-8),
            0
        )
    )
    
    # Calculate True Range Sum (10-day)
    data['true_range_sum'] = data['true_range'].rolling(window=10, min_periods=5).sum()
    
    # Compute Volume-Weighted Efficiency
    data['raw_efficiency'] = data['net_price_change'] / (data['true_range_sum'] + 1e-8)
    data['volume_weighted_efficiency'] = data['raw_efficiency'] * (1 + data['volume_skew'])
    
    # Apply Directional Filter
    data['trend_short'] = data['close'] - data['close'].shift(5)
    data['trend_medium'] = data['close'].shift(5) - data['close'].shift(10)
    
    # Check trend alignment
    data['trend_aligned'] = np.sign(data['trend_short']) == np.sign(data['trend_medium'])
    data['direction_filter'] = np.where(data['trend_aligned'], 1, -1)
    data['filtered_efficiency'] = data['volume_weighted_efficiency'] * data['direction_filter']
    
    # Smooth with Adaptive Window
    data['recent_volatility'] = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    data['volatility_rank'] = data['recent_volatility'].rolling(window=20, min_periods=10).rank(pct=True)
    
    # Adaptive smoothing period based on volatility
    data['smooth_window'] = np.where(
        data['volatility_rank'] > 0.7, 
        3,  # Shorter window for high volatility
        np.where(data['volatility_rank'] < 0.3, 10, 5)  # Longer window for low volatility
    )
    
    # Apply adaptive smoothing
    factor_values = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i < 20:  # Minimum periods for calculations
            factor_values.iloc[i] = 0
            continue
            
        current_window = int(data['smooth_window'].iloc[i])
        start_idx = max(0, i - current_window + 1)
        factor_values.iloc[i] = data['filtered_efficiency'].iloc[start_idx:i+1].mean()
    
    # Clean up and return
    factor_values = factor_values.replace([np.inf, -np.inf], 0).fillna(0)
    return factor_values
