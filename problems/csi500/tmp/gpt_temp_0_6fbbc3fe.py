import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate all required components
    # Momentum Convergence-Divergence with Volume Validation
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_8d'] = data['close'] / data['close'].shift(8) - 1
    data['mom_13d'] = data['close'] / data['close'].shift(13) - 1
    
    data['short_term_diff'] = data['mom_3d'] - data['mom_8d']
    data['long_term_diff'] = data['mom_8d'] - data['mom_13d']
    data['convergence_score'] = data['short_term_diff'] * data['long_term_diff']
    
    data['vol_mom_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['vol_mom_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['vol_mom_ratio'] = data['vol_mom_5d'] / (data['vol_mom_10d'] + 1e-8)
    data['signal_1'] = data['convergence_score'] * data['vol_mom_ratio']
    
    # Intraday Pressure with Acceleration Confirmation
    data['norm_close_pos'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['pressure_strength'] = 2 * data['norm_close_pos'] - 1
    
    data['ret_2d'] = data['close'] / data['close'].shift(2) - 1
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['acceleration'] = data['ret_5d'] - data['ret_2d']
    
    data['base_signal'] = data['pressure_strength'] * data['acceleration']
    data['vol_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['signal_2'] = data['base_signal'] * (data['volume'] / data['vol_5d_avg'])
    
    # Relative Strength Breakout with Volume Divergence
    data['high_20d'] = data['close'].rolling(window=20).max()
    data['strength_ratio'] = data['close'] / data['high_20d']
    
    data['vol_div_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['vol_div_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_divergence'] = data['vol_div_5d'] - data['vol_div_10d']
    
    data['breakout_condition'] = ((data['strength_ratio'] > 0.95) & (data['volume_divergence'] > 0)).astype(float)
    data['gap_enhancement'] = data['open'] / data['close'].shift(1) - 1
    data['signal_3'] = data['breakout_condition'] * data['gap_enhancement']
    
    # Volatility-Adjusted Turnover Efficiency
    data['avg_price'] = data['amount'] / (data['volume'] + 1e-8)
    data['turnover'] = data['volume'] * data['avg_price']
    
    # Calculate 10-day path length
    data['daily_ret'] = data['close'] / data['close'].shift(1) - 1
    data['path_length'] = data['daily_ret'].abs().rolling(window=10).sum()
    data['net_return_10d'] = data['close'] / data['close'].shift(10) - 1
    data['efficiency_ratio'] = data['net_return_10d'] / (data['path_length'] + 1e-8)
    
    data['base_factor'] = data['efficiency_ratio'] * data['turnover']
    data['vol_filter'] = data['volume'] / data['volume'].shift(5) - 1
    data['signal_4'] = data['base_factor'] * data['vol_filter']
    
    # Gap Filling with Momentum Persistence
    data['gap_size'] = data['open'] / data['close'].shift(1) - 1
    data['gap_direction'] = np.sign(data['gap_size'])
    
    # Calculate momentum persistence
    data['daily_ret_sign'] = np.sign(data['daily_ret'])
    data['persistence_strength'] = 0
    
    for i in range(len(data)):
        if i >= 3:
            current_sign = data['daily_ret_sign'].iloc[i]
            prev_sign_1 = data['daily_ret_sign'].iloc[i-1]
            prev_sign_2 = data['daily_ret_sign'].iloc[i-2]
            
            if current_sign == prev_sign_1 == prev_sign_2:
                data.loc[data.index[i], 'persistence_strength'] = 3
            elif current_sign == prev_sign_1:
                data.loc[data.index[i], 'persistence_strength'] = 2
            else:
                data.loc[data.index[i], 'persistence_strength'] = 1
    
    data['base_probability'] = data['gap_size'] * data['persistence_strength']
    data['vol_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['signal_5'] = data['base_probability'] * (data['volume'] / data['vol_20d_avg'])
    
    # Combine all signals with equal weights
    signals = ['signal_1', 'signal_2', 'signal_3', 'signal_4', 'signal_5']
    valid_signals = [col for col in signals if col in data.columns]
    
    if valid_signals:
        combined_signal = data[valid_signals].mean(axis=1)
    else:
        combined_signal = pd.Series(0, index=data.index)
    
    # Fill NaN values with 0
    result = combined_signal.fillna(0)
    
    return result
