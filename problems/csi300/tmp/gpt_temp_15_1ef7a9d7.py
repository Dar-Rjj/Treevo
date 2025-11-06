import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns and volume changes
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    data['vol_change'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Directional Volume-Price Correlation
    # Up-day correlation
    up_mask = data['close'] > data['open']
    up_corr_values = []
    for i in range(len(data)):
        if i < 3:
            up_corr_values.append(np.nan)
            continue
        window_data = data.iloc[i-2:i+1]  # 3-day window
        up_window = window_data[window_data['close'] > window_data['open']]
        if len(up_window) >= 2:
            corr = up_window['ret'].corr(up_window['vol_change'])
            up_corr_values.append(corr if not np.isnan(corr) else 0)
        else:
            up_corr_values.append(0)
    data['up_corr'] = up_corr_values
    
    # Down-day correlation
    down_mask = data['close'] < data['open']
    down_corr_values = []
    for i in range(len(data)):
        if i < 3:
            down_corr_values.append(np.nan)
            continue
        window_data = data.iloc[i-2:i+1]  # 3-day window
        down_window = window_data[window_data['close'] < window_data['open']]
        if len(down_window) >= 2:
            corr = down_window['ret'].corr(down_window['vol_change'])
            down_corr_values.append(corr if not np.isnan(corr) else 0)
        else:
            down_corr_values.append(0)
    data['down_corr'] = down_corr_values
    
    # Correlation asymmetry
    data['corr_asymmetry'] = np.where(
        data['down_corr'] != 0, 
        data['up_corr'] / data['down_corr'], 
        1.0
    )
    
    # Directional Price Efficiency
    data['efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['up_efficiency'] = np.where(up_mask, data['efficiency'], np.nan)
    data['down_efficiency'] = np.where(down_mask, data['efficiency'], np.nan)
    
    # Rolling averages for efficiency
    data['up_eff_avg'] = data['up_efficiency'].rolling(window=5, min_periods=1).mean()
    data['down_eff_avg'] = data['down_efficiency'].rolling(window=5, min_periods=1).mean()
    
    # Efficiency asymmetry
    data['eff_asymmetry'] = np.where(
        data['down_eff_avg'] != 0, 
        data['up_eff_avg'] / data['down_eff_avg'], 
        1.0
    )
    
    # Directional Breakout Strength
    # Calculate rolling highs and lows
    data['high_3d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    
    # Up-breakout
    data['up_breakout'] = np.where(
        up_mask,
        (data['high'] - data['high_3d'].shift(1)) / (data['high'] - data['low']),
        0
    )
    
    # Down-breakout
    data['down_breakout'] = np.where(
        down_mask,
        (data['low_3d'].shift(1) - data['low']) / (data['high'] - data['low']),
        0
    )
    
    # Rolling averages for breakout
    data['up_breakout_avg'] = data['up_breakout'].rolling(window=5, min_periods=1).mean()
    data['down_breakout_avg'] = data['down_breakout'].rolling(window=5, min_periods=1).mean()
    
    # Breakout asymmetry
    data['breakout_asymmetry'] = np.where(
        data['down_breakout_avg'] != 0, 
        data['up_breakout_avg'] / data['down_breakout_avg'], 
        1.0
    )
    
    # Directional Momentum with Volume
    data['price_change'] = data['close'] - data['close'].shift(1)
    
    # Up-momentum
    data['up_momentum'] = np.where(
        up_mask,
        data['price_change'] * data['volume'] * data['up_corr'],
        0
    )
    
    # Down-momentum
    data['down_momentum'] = np.where(
        down_mask,
        data['price_change'] * data['volume'] * data['down_corr'],
        0
    )
    
    # Rolling averages for momentum
    data['up_momentum_avg'] = data['up_momentum'].rolling(window=5, min_periods=1).mean()
    data['down_momentum_avg'] = data['down_momentum'].rolling(window=5, min_periods=1).mean()
    
    # Momentum asymmetry
    data['momentum_asymmetry'] = np.where(
        data['down_momentum_avg'] != 0, 
        data['up_momentum_avg'] / data['down_momentum_avg'], 
        1.0
    )
    
    # Combined Asymmetric Alpha
    data['core_asymmetry'] = (
        data['corr_asymmetry'] * 
        data['eff_asymmetry'] * 
        data['breakout_asymmetry']
    )
    
    data['final_alpha'] = data['core_asymmetry'] * data['momentum_asymmetry']
    
    # Handle infinite values and normalize
    data['final_alpha'] = data['final_alpha'].replace([np.inf, -np.inf], np.nan)
    data['final_alpha'] = (data['final_alpha'] - data['final_alpha'].mean()) / data['final_alpha'].std()
    
    return data['final_alpha']
