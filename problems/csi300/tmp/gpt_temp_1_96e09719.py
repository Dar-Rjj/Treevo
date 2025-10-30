import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate True Range for volatility
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility-Adjusted Reversal Component
    # Recent Price Extremes
    data['highest_10d'] = data['high'].rolling(window=10, min_periods=5).max()
    data['lowest_10d'] = data['low'].rolling(window=10, min_periods=5).min()
    
    # Reversal Strength
    data['dist_from_high'] = (data['highest_10d'] - data['close']) / data['highest_10d']
    data['dist_from_low'] = (data['close'] - data['lowest_10d']) / data['lowest_10d']
    
    # Weight by volatility (inverse relationship)
    data['volatility_weight'] = 1 / (1 + data['true_range'].rolling(window=10, min_periods=5).mean())
    data['reversal_signal'] = np.where(
        data['dist_from_high'] > data['dist_from_low'],
        data['dist_from_high'] * data['volatility_weight'],
        -data['dist_from_low'] * data['volatility_weight']
    )
    
    # Momentum Confirmation Component
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Exponential decay weights
    decay_5d = np.exp(-np.arange(5)[::-1] / 2.5)
    decay_20d = np.exp(-np.arange(20)[::-1] / 5.0)
    
    # Apply exponential decay to momentum
    data['momentum_5d_ewma'] = data['momentum_5d'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x * decay_5d[-len(x):]) / np.sum(decay_5d[-len(x):]), raw=False
    )
    data['momentum_20d_ewma'] = data['momentum_20d'].rolling(window=20, min_periods=10).apply(
        lambda x: np.sum(x * decay_20d[-len(x):]) / np.sum(decay_20d[-len(x):]), raw=False
    )
    
    # Combined momentum signal
    data['momentum_signal'] = 0.6 * data['momentum_5d_ewma'] + 0.4 * data['momentum_20d_ewma']
    
    # Volume-Based Signal Enhancement
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_surge'] = np.where(
        data['volume'] > 1.5 * data['volume_20d_avg'], 1, 0
    )
    
    # Multi-Timeframe Integration
    # Combine reversal and momentum with volume filter
    data['raw_factor'] = data['reversal_signal'] * data['momentum_signal']
    
    # Apply volume surge as confirmation
    data['volume_enhanced'] = data['raw_factor'] * (1 + 0.3 * data['volume_surge'])
    
    # Final volatility weighting
    vol_ma = data['true_range'].rolling(window=20, min_periods=10).mean()
    data['final_factor'] = data['volume_enhanced'] / (1 + vol_ma)
    
    # Clean up intermediate columns
    result = data['final_factor'].copy()
    
    return result
