import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate returns
    data['return_1d'] = data['close'].pct_change()
    
    # Momentum Divergence
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_divergence'] = abs(data['momentum_5d'] - data['momentum_20d'])
    
    # Volume-Price Efficiency
    # Intraday efficiency
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_efficiency'] = data['intraday_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume asymmetry
    data['is_up_day'] = (data['close'] > data['open']).astype(int)
    data['is_down_day'] = (data['close'] < data['open']).astype(int)
    
    # Calculate rolling sums for up and down days
    data['up_volume_5d'] = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x * data.loc[x.index, 'is_up_day']), raw=False
    )
    data['down_volume_5d'] = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x * data.loc[x.index, 'is_down_day']), raw=False
    )
    
    # Volume asymmetry ratio
    data['volume_asymmetry'] = data['up_volume_5d'] / data['down_volume_5d']
    data['volume_asymmetry'] = data['volume_asymmetry'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-weighted return
    def calc_volume_weighted_return(window):
        returns = data.loc[window.index, 'return_1d']
        volumes = window.values
        return np.sum(returns * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else 0
    
    data['volume_weighted_return'] = data['volume'].rolling(window=5).apply(
        calc_volume_weighted_return, raw=False
    )
    
    # Volatility Adjustment
    # True Range calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Recent volatility (10-day)
    data['recent_volatility'] = data['true_range'].rolling(window=10).std()
    
    # Historical volatility (60-day)
    data['historical_volatility'] = data['true_range'].rolling(window=60).std()
    
    # Volatility ratio
    data['volatility_ratio'] = data['recent_volatility'] / data['historical_volatility']
    data['volatility_ratio'] = data['volatility_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Combined Alpha Factor
    # Base factor
    data['base_factor'] = (data['momentum_divergence'] * 
                          data['intraday_efficiency'] * 
                          data['volume_asymmetry'])
    
    # Volume confirmation
    raw_5d_return = data['close'] / data['close'].shift(5) - 1
    data['volume_confirmation'] = data['base_factor'] * (data['volume_weighted_return'] / raw_5d_return)
    data['volume_confirmation'] = data['volume_confirmation'].replace([np.inf, -np.inf], np.nan)
    
    # Volatility scaling
    def volatility_scaling(vol_ratio):
        if vol_ratio < 0.8:
            return 1.3
        elif vol_ratio > 1.2:
            return 0.7
        else:
            return 1.0
    
    data['volatility_scaling'] = data['volatility_ratio'].apply(volatility_scaling)
    
    # Final alpha factor
    data['alpha_factor'] = data['volume_confirmation'] * data['volatility_scaling']
    
    return data['alpha_factor']
