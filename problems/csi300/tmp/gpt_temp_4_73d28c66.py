import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Dynamic Liquidity Absorption Factor
    upside_absorption = (data['high'] - data['open']) * data['volume']
    downside_absorption = (data['open'] - data['low']) * data['volume']
    net_absorption = upside_absorption - downside_absorption
    total_absorption = upside_absorption + downside_absorption
    
    # Calculate 5-day close price slope
    close_slope = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            window_data = data['close'].iloc[i-5:i+1]
            if len(window_data) == 6:
                x = np.arange(len(window_data))
                slope, _, _, _, _ = linregress(x, window_data.values)
                close_slope.iloc[i] = slope
            else:
                close_slope.iloc[i] = 0
        else:
            close_slope.iloc[i] = 0
    
    absorption_factor = (net_absorption / total_absorption.replace(0, 1)) * close_slope
    
    # Volatility Clustering Breakout Detector
    short_vol = data['high'].rolling(window=6, min_periods=1).apply(lambda x: (x.max() - x.min()) if len(x) == 6 else 0)
    medium_vol = data['high'].rolling(window=21, min_periods=1).apply(lambda x: (x.max() - x.min()) if len(x) == 21 else 0)
    volatility_ratio = short_vol / medium_vol.replace(0, 1)
    
    bar_strength = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1)
    volume_ratio = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    
    breakout_signal = volatility_ratio * bar_strength * volume_ratio * np.sign(data['close'] - data['open'])
    
    # Momentum Divergence Oscillator
    fast_momentum = data['close'] / data['close'].shift(2) - 1
    slow_momentum = data['close'] / data['close'].shift(9) - 1
    momentum_spread = fast_momentum - slow_momentum
    
    direction_indicator = np.sign(fast_momentum * slow_momentum)
    
    # Calculate 5-day volume slope
    volume_slope = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            window_data = data['volume'].iloc[i-5:i+1]
            if len(window_data) == 6:
                x = np.arange(len(window_data))
                slope, _, _, _, _ = linregress(x, window_data.values)
                volume_slope.iloc[i] = slope
            else:
                volume_slope.iloc[i] = 0
        else:
            volume_slope.iloc[i] = 0
    
    momentum_oscillator = momentum_spread * direction_indicator * volume_slope
    
    # Price-Volume Congestion Breakout
    # Calculate ATR_10 and ATR_20
    tr = pd.DataFrame(index=data.index)
    tr['h_l'] = data['high'] - data['low']
    tr['h_pc'] = abs(data['high'] - data['close'].shift(1))
    tr['l_pc'] = abs(data['low'] - data['close'].shift(1))
    true_range = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    
    atr_10 = true_range.rolling(window=10, min_periods=1).mean()
    atr_20 = true_range.rolling(window=20, min_periods=1).mean()
    
    price_compression = atr_10 / atr_20.replace(0, 1)
    volume_drying = data['volume'] / data['volume'].rolling(window=10, min_periods=1).max()
    price_breakout = (data['high'] - data['low']) / atr_10.replace(0, 1)
    volume_expansion = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    
    position_ratio = 2 * (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, 1) - 1
    
    congestion_breakout = price_breakout * volume_expansion * position_ratio
    
    # Combine all factors with equal weights
    final_factor = (
        absorption_factor.fillna(0) + 
        breakout_signal.fillna(0) + 
        momentum_oscillator.fillna(0) + 
        congestion_breakout.fillna(0)
    )
    
    return final_factor
