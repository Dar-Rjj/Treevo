import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Components
    # Volatility-Adjusted Return
    data['vol_adj_return'] = ((data['close'] / data['close'].shift(1) - 1) / 
                             (data['high'] - data['low']) / data['close'])
    
    # Price Efficiency
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Volatility Persistence
    data['vol_persistence'] = ((data['high'] - data['low']) / data['close']) / \
                             ((data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1))
    
    # Volume-Price Components
    # Volume Efficiency
    data['volume_efficiency'] = (data['close'] - data['open']) / data['volume']
    
    # Volume-Volatility Alignment
    vol_change = (data['high'] - data['low']) / data['close'] - \
                (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1)
    volume_change = data['volume'] / data['volume'].shift(1) - 1
    data['vol_vol_alignment'] = np.sign(volume_change) * np.sign(vol_change)
    
    # Regime-Asymmetric Dynamics
    # High Volatility Factor
    data['high_vol_factor'] = (data['volume'] / data['volume'].shift(1)) * \
                             ((data['close'] - data['open']) / (data['high'] - data['low']))
    
    # Low Volatility Factor
    data['low_vol_factor'] = (data['volume'] / data['amount']) * \
                            (np.abs(data['close'] - data['open']) / data['volume'])
    
    # Momentum Integration
    # Volatility-Weighted Momentum
    vol_ratio = (data['high'] - data['low']) / data['close']
    vol_ma = vol_ratio.rolling(window=5, min_periods=3).mean()
    data['vol_weighted_momentum'] = data['vol_adj_return'] * np.sign(vol_ratio - vol_ma)
    
    # Volume-Confirmed Momentum
    def calc_avg_volatility(window):
        return np.mean((window['high'] - window['low']) / window['close'])
    
    # Calculate 3-day momentum with volatility normalization
    momentum_3d = (data['close'] / data['close'].shift(3) - 1)
    
    # Calculate average volatility over t-2 to t period
    vol_window_avg = []
    for i in range(len(data)):
        if i >= 2:
            window_data = data.iloc[i-2:i+1]
            avg_vol = calc_avg_volatility(window_data)
        else:
            avg_vol = np.nan
        vol_window_avg.append(avg_vol)
    
    data['avg_vol_3d'] = vol_window_avg
    volume_change_sign = np.sign(data['volume'] / data['volume'].shift(1) - 1)
    data['volume_confirmed_momentum'] = (momentum_3d / data['avg_vol_3d']) * volume_change_sign
    
    # Alpha Construction
    # Core Divergence
    data['core_divergence'] = (data['high_vol_factor'] + data['low_vol_factor']) * data['vol_persistence']
    
    # Final Alpha
    alpha = data['core_divergence'] * data['vol_weighted_momentum'] * data['volume_confirmed_momentum']
    
    return alpha
