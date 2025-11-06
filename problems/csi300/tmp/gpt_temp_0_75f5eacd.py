import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Volatility Regime Classification
    # Range Volatility Ratio
    data['range_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['range_20d'] = (data['high'] - data['low']).rolling(window=20).mean()
    data['range_vol_ratio'] = data['range_5d'] / data['range_20d']
    
    # Volatility threshold using returns standard deviation
    data['returns_std_10d'] = data['returns'].rolling(window=10).std()
    data['vol_threshold'] = data['returns_std_10d'] * data['close']
    
    # Regime definition
    data['high_vol_regime'] = ((data['range_vol_ratio'] > 1.2) | 
                              ((data['high'] - data['low']) > data['vol_threshold']))
    data['low_vol_regime'] = ((data['range_vol_ratio'] <= 1.2) & 
                             ((data['high'] - data['low']) <= data['vol_threshold']))
    
    # Price Acceleration Analysis
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['acceleration'] = data['momentum_5d'] / data['momentum_5d'].shift(3) - 1
    
    # Volume Asymmetry Analysis
    # Create boolean masks for up and down days
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    # Calculate volume sums for up and down days using rolling apply
    def sum_volume_up(window):
        return window[up_days.loc[window.index]].sum()
    
    def sum_volume_down(window):
        return window[down_days.loc[window.index]].sum()
    
    data['volume_up_sum'] = data['volume'].rolling(window=5).apply(sum_volume_up, raw=False)
    data['volume_down_sum'] = data['volume'].rolling(window=5).apply(sum_volume_down, raw=False)
    data['volume_asymmetry_ratio'] = data['volume_up_sum'] / data['volume_down_sum']
    
    # Volume Momentum
    data['volume_ma_10d'] = data['volume'].rolling(window=10).mean()
    data['volume_momentum'] = data['volume'] / data['volume_ma_10d']
    
    # Range Efficiency Analysis
    data['efficiency_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Regime-Adaptive Signal Construction
    # High Volatility Regime Signal
    data['returns_std_20d'] = data['returns'].rolling(window=20).std()
    
    # Acceleration-Volume Divergence
    data['price_accel_norm'] = data['acceleration'] / data['returns_std_20d']
    data['volume_momentum_norm'] = data['volume_momentum'] / data['returns_std_20d']
    
    # Divergence Signal
    data['divergence_signal'] = (np.sign(data['price_accel_norm']) * 
                                np.sign(data['volume_momentum_norm']) * 
                                (abs(data['price_accel_norm']) - abs(data['volume_momentum_norm'])))
    
    # Volume Asymmetry Confirmation
    data['high_vol_signal'] = (data['divergence_signal'] * 
                              (data['volume_asymmetry_ratio'] - 1) * 
                              data['efficiency_ratio'])
    
    # Low Volatility Regime Signal
    # Acceleration-Adjusted Momentum
    data['raw_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['acceleration_factor'] = 1 + data['acceleration']
    
    # Volume Confirmation
    data['momentum_volume'] = data['raw_momentum'] * data['acceleration_factor'] * data['volume_momentum']
    
    # Efficiency Filter
    data['abs_returns_10d'] = abs(data['close'] - data['close'].shift(10))
    data['sum_abs_returns_10d'] = abs(data['close'] - data['close'].shift(1)).rolling(window=10).sum()
    data['efficiency_filter'] = data['abs_returns_10d'] / data['sum_abs_returns_10d']
    data['efficiency_filter'] = data['efficiency_filter'].replace([np.inf, -np.inf], np.nan)
    
    data['low_vol_signal'] = data['momentum_volume'] * data['efficiency_ratio'] * data['efficiency_filter']
    
    # Final Alpha Signal - Regime-Specific Selection
    data['regime_signal'] = np.where(data['high_vol_regime'], 
                                   data['high_vol_signal'], 
                                   np.where(data['low_vol_regime'], 
                                           data['low_vol_signal'], 0))
    
    # Volatility Scaling
    data['returns_std_20d_annualized'] = data['returns_std_20d'] * np.sqrt(252)
    data['final_signal'] = data['regime_signal'] / data['returns_std_20d_annualized']
    
    # Clean up intermediate columns
    result = data['final_signal'].copy()
    
    return result
