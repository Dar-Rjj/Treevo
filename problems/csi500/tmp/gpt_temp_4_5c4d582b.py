import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Acceleration Component
    # First-order momentum
    data['momentum_1'] = data['close'] / data['close'].shift(1) - 1
    
    # Second-order acceleration
    data['acceleration'] = data['momentum_1'] - data['momentum_1'].shift(1)
    
    # Acceleration magnitude and direction
    data['accel_magnitude'] = data['acceleration'].abs()
    data['accel_direction'] = np.sign(data['acceleration'])
    
    # Volume-Based Weighting
    # Volume moving averages
    short_vol_ma = 5
    long_vol_ma = 20
    data['vol_ma_short'] = data['volume'].rolling(window=short_vol_ma, min_periods=1).mean()
    data['vol_ma_long'] = data['volume'].rolling(window=long_vol_ma, min_periods=1).mean()
    
    # Volume trend strength
    data['vol_ratio'] = data['vol_ma_short'] / data['vol_ma_long']
    data['vol_trend_strength'] = np.log(data['vol_ratio'])
    
    # Volume trend consistency (using rolling standard deviation)
    data['vol_consistency'] = 1 / (1 + data['volume'].rolling(window=10, min_periods=1).std())
    
    # Apply volume weight to acceleration
    data['volume_weight'] = data['vol_trend_strength'] * data['vol_consistency']
    data['weighted_acceleration'] = data['acceleration'] * data['volume_weight']
    
    # Liquidity Adjustment
    # Calculate turnover-based liquidity (assuming amount represents turnover)
    data['turnover'] = data['amount'] / data['close']
    data['avg_turnover'] = data['turnover'].rolling(window=20, min_periods=1).mean()
    
    # Relative liquidity score
    data['liquidity_score'] = data['turnover'] / data['avg_turnover']
    
    # Liquidity stability
    data['liquidity_stability'] = 1 / (1 + data['turnover'].rolling(window=10, min_periods=1).std())
    
    # Combine acceleration with liquidity
    data['liquidity_adjusted_accel'] = data['weighted_acceleration'] * data['liquidity_score'] * data['liquidity_stability']
    
    # Apply time decay to recent signals (exponential weighting)
    decay_window = 10
    weights = np.exp(-np.arange(decay_window) / decay_window)
    weights = weights / weights.sum()
    
    # Create final alpha factor with time decay
    data['alpha_factor'] = data['liquidity_adjusted_accel'].rolling(
        window=decay_window, min_periods=1
    ).apply(lambda x: np.sum(x * weights[-len(x):]) if len(x) > 0 else np.nan)
    
    # Return the alpha factor series
    return data['alpha_factor']
