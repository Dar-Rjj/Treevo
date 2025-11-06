import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Volatility Structure components
    data['range_efficiency'] = ((data['high'] - data['low']) / (np.abs(data['open'] - data['prev_close']) + 1e-8) * 
                               (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * 
                               np.sign(data['volume'] - data['prev_volume']))
    
    data['gap_volatility'] = (np.abs(data['open'] - data['prev_close']) / (data['prev_high'] - data['prev_low'] + 1e-8) * 
                             np.log(data['high'] - data['low'] + 1) / (np.log(np.abs(data['open'] - data['prev_close']) + 1) + 1e-8) * 
                             np.abs(data['volume'] - data['prev_volume']) / (data['prev_volume'] + 1e-8))
    
    # Volatility Persistence
    vol_increase_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:
            count = 0
            for j in range(i-2, i+1):
                if j > 0 and (data['high'].iloc[j] - data['low'].iloc[j]) > (data['high'].iloc[j-1] - data['low'].iloc[j-1]):
                    count += 1
            vol_increase_count.iloc[i] = count
        else:
            vol_increase_count.iloc[i] = 0
    
    data['volatility_persistence'] = (np.sign(data['close'] - data['open']) * 
                                     np.sign(data['close'].shift(1) - data['open'].shift(1)) * 
                                     vol_increase_count * 
                                     (data['amount'] / data['prev_amount'] - 1))
    
    # Liquidity Microstructure
    data['volume_to_range'] = (data['volume'] / (data['high'] - data['low'] + 1e-8) * 
                              np.abs(data['close'] - data['open']) / (np.abs(data['open'] - data['prev_close']) + 1e-8) * 
                              (data['close'] - data['prev_close']) / (np.abs(data['close'] - data['prev_close']) + 1e-8))
    
    data['position_bias'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8) - 0.5 * 
                            np.abs(data['open'] - data['prev_close']) / (data['high'] - data['low'] + 1e-8) * 
                            data['volume'] / (data['prev_volume'] + 1e-8))
    
    data['opening_pressure'] = ((data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8) * 
                               np.abs(data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8) * 
                               (data['amount'] / data['prev_amount'] - 1))
    
    # Momentum Framework
    data['volatility_weighted'] = ((data['close'] - data['close'].shift(2)) / (data['high'] - data['low'] + 1e-8) * 
                                  np.abs(data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8) * 
                                  np.sign(data['volume'] - data['prev_volume']))
    
    # Volume-Confirmed momentum
    close_diff_sum = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            diff_sum = 0
            for j in range(i-4, i+1):
                if j > 0:
                    diff_sum += np.abs(data['close'].iloc[j] - data['close'].iloc[j-1])
            close_diff_sum.iloc[i] = diff_sum
        else:
            close_diff_sum.iloc[i] = 0
    
    data['volume_confirmed'] = ((data['close'] - data['close'].shift(5)) / (close_diff_sum + 1e-8) * 
                               data['volume'] / (data['prev_volume'] + 1e-8) * 
                               np.abs(data['open'] - data['prev_close']) / (data['high'] - data['low'] + 1e-8))
    
    data['amount_efficient'] = (((data['close'] / data['prev_close'] - 1) * 
                                (data['prev_close'] / data['close'].shift(2) - 1)) / 
                               (np.abs((data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)) + 1e-8) * 
                               data['amount'] / (data['prev_amount'] + 1e-8))
    
    # Entropy Dynamics
    price_ratio = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    price_ratio = np.clip(price_ratio, 1e-8, 1-1e-8)
    data['price_entropy'] = (-price_ratio * np.log(price_ratio) * 
                            np.abs(data['close'] - data['open']) / (np.abs(data['open'] - data['prev_close']) + 1e-8) * 
                            data['volume'] / (data['prev_volume'] + 1e-8))
    
    # Volume Entropy
    volume_entropy_vals = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            vol_window = data['volume'].iloc[i-4:i+1]
            vol_min = vol_window.min()
            vol_max = vol_window.max()
            if vol_max > vol_min:
                vol_ratio = (data['volume'].iloc[i] - vol_min) / (vol_max - vol_min)
                vol_ratio = np.clip(vol_ratio, 1e-8, 1-1e-8)
                volume_entropy_vals.iloc[i] = -vol_ratio * np.log(vol_ratio) * data['volume'].iloc[i] * (data['amount'].iloc[i] / data['prev_amount'].iloc[i] - 1)
            else:
                volume_entropy_vals.iloc[i] = 0
        else:
            volume_entropy_vals.iloc[i] = 0
    
    data['volume_entropy'] = volume_entropy_vals
    
    # Range Entropy
    range_avg = data['high'].rolling(window=5, min_periods=1).mean() - data['low'].rolling(window=5, min_periods=1).mean()
    data['range_entropy'] = ((data['high'] - data['low']) / (range_avg + 1e-8) * 
                            np.abs(data['close'] - data['open']) / (np.abs(data['open'] - data['prev_close']) + 1e-8) * 
                            np.abs(data['volume'] - data['prev_volume']) / (data['prev_volume'] + 1e-8))
    
    # Breakout Integration
    data['volume_breakout_signal'] = ((data['volume'] > 1.5 * data['prev_volume']) & 
                                     (data['position_bias'] > 0.3 * np.abs(data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8))).astype(float)
    
    data['efficiency_breakout_signal'] = ((data['amount'] / (data['volume'] * data['close'] + 1e-8) > 1.5) & 
                                         (data['opening_pressure'] > 0.6 * data['volume'] / (data['prev_volume'] + 1e-8))).astype(float)
    
    data['breakout_momentum'] = (data['volatility_weighted'] * 
                                (1 + data['volume_breakout_signal'] + data['efficiency_breakout_signal']) * 
                                np.abs(data['volume'] - data['prev_volume']) / (data['prev_volume'] + 1e-8))
    
    # Asymmetry Dynamics
    data['fractal_convergence'] = (data['price_entropy'] * data['volume_entropy'] * 
                                  (data['close'] - data['prev_close']) * 
                                  np.sign(data['volume'] - data['prev_volume']))
    
    data['fractal_gradient'] = ((data['price_entropy'] - data['volume_entropy']) * 
                               (data['volume_entropy'] - data['range_entropy']) * 
                               data['volume'] * 
                               (data['amount'] / data['prev_amount'] - 1))
    
    data['fractal_divergence'] = (np.abs(data['price_entropy'] - data['volume_entropy']) / 
                                 (data['price_entropy'] + data['volume_entropy'] + 1e-8) * 
                                 (data['volume'] - data['prev_volume']) * 
                                 np.sign(data['close'] - data['prev_close']))
    
    # Efficiency Measures
    data['fractal_efficiency'] = (data['volume'] / (np.abs(data['close'] - data['prev_close']) + 1e-8) * 
                                 np.abs(data['close'] - data['open']) / (np.abs(data['open'] - data['prev_close']) + 1e-8) * 
                                 (data['amount'] / data['prev_amount'] - 1))
    
    data['fractal_utilization'] = (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * 
                                  (data['close'] - data['prev_close']) * 
                                  np.sign(data['volume'] - data['prev_volume']))
    
    data['fractal_discovery'] = (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * 
                                (data['close'] - data['prev_close']) / (np.abs(data['close'] - data['prev_close']) + 1e-8) * 
                                data['volume'] / (data['prev_volume'] + 1e-8))
    
    # Momentum Asymmetry
    data['upward_fracture'] = np.where(data['close'] > data['prev_close'], 
                                      (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8), 
                                      0) * data['volume'] / (data['prev_volume'] + 1e-8)
    
    data['downward_fracture'] = np.where(data['close'] < data['prev_close'], 
                                        (data['close'] - data['low']) / (data['high'] - data['close'] + 1e-8), 
                                        0) * data['volume'] / (data['prev_volume'] + 1e-8)
    
    data['momentum_asymmetry'] = (data['upward_fracture'] - data['downward_fracture']) * np.abs(data['close'] - data['open']) / (np.abs(data['open'] - data['prev_close']) + 1e-8)
    
    # Transition Framework
    data['imbalance_transition'] = data['opening_pressure'] - data['fractal_discovery'] * data['volume'] / (data['prev_volume'] + 1e-8)
    data['efficiency_transition'] = data['fractal_efficiency'] - data['volume_to_range'] * np.abs(data['volume'] - data['prev_volume']) / (data['prev_volume'] + 1e-8)
    data['momentum_transition'] = data['upward_fracture'] - data['downward_fracture'] * data['amount'] / (data['prev_amount'] + 1e-8)
    
    # Alpha Synthesis
    micro_level = data['range_efficiency'] * data['price_entropy'] * np.sign(data['volume'] - data['prev_volume'])
    meso_level = data['volume_to_range'] * data['volume_entropy'] * (data['amount'] / data['prev_amount'] - 1)
    macro_level = data['volatility_weighted'] * data['range_entropy'] * data['volume'] / (data['prev_volume'] + 1e-8)
    asymmetry_core = data['momentum_asymmetry'] * data['fractal_divergence'] * np.sign(data['close'] - data['prev_close'])
    breakout_filter = data['breakout_momentum'] * data['fractal_discovery'] * np.abs(data['volume'] - data['prev_volume']) / (data['prev_volume'] + 1e-8)
    
    # Final Alpha
    final_alpha = (micro_level + meso_level + macro_level) * asymmetry_core * breakout_filter
    
    # Clean up and return
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    return final_alpha
