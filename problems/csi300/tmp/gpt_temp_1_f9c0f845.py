import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate basic components with error handling for division by zero
    def safe_divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    
    # Multi-Timeframe Volatility Regime Detection
    data['intraday_vol_regime'] = safe_divide(data['high'] - data['low'], data['close'].shift(1)) * \
                                 safe_divide(data['volume'], data['volume'].shift(1)) * \
                                 np.sign(data['close'] - data['open'])
    
    # Short-term volatility persistence
    vol_persistence = []
    for i in range(len(data)):
        if i >= 2:
            vol_sum = 0
            count = 0
            for j in range(i-2, i+1):
                if j >= 0 and j-1 >= 0 and data['close'].iloc[j-1] != 0:
                    vol_sum += safe_divide(data['high'].iloc[j] - data['low'].iloc[j], data['close'].iloc[j-1])
                    count += 1
            if count > 0:
                vol_persistence.append(vol_sum / count * 
                                     safe_divide(data['volume'].iloc[i], data['volume'].iloc[i-2]) * 
                                     np.sign(data['close'].iloc[i] - data['close'].iloc[i-2]))
            else:
                vol_persistence.append(0)
        else:
            vol_persistence.append(0)
    data['short_term_vol_persistence'] = vol_persistence
    
    # Medium-term volatility expansion
    data['medium_term_vol_expansion'] = safe_divide(data['high'] - data['low'], 
                                                   (data['high'].shift(5) - data['low'].shift(5))) * \
                                       safe_divide(data['volume'], data['volume'].shift(5)) * \
                                       np.sign(data['close'] - data['close'].shift(5))
    
    # Volume-Volatility Asymmetry Analysis
    hl_range = data['high'] - data['low']
    data['high_side_vol_pressure'] = safe_divide(data['close'] - data['low'], hl_range) * \
                                    safe_divide(data['volume'], data['amount']) * \
                                    np.sign(data['close'] - data['open'])
    
    data['low_side_vol_pressure'] = safe_divide(data['high'] - data['close'], hl_range) * \
                                   safe_divide(data['volume'], data['amount']) * \
                                   np.sign(data['close'] - data['open']) * -1
    
    data['vol_vol_divergence'] = (data['high_side_vol_pressure'] - data['low_side_vol_pressure']) * \
                                safe_divide(hl_range, data['close'].shift(1)) * \
                                safe_divide(data['volume'], data['volume'].shift(1))
    
    # Price-Volume Fractal Dynamics
    data['vol_weighted_price_fractal'] = safe_divide(data['close'] - data['open'], hl_range) * \
                                        safe_divide(data['volume'], data['amount']) * \
                                        np.abs(data['close'] - data['close'].shift(1)) / hl_range
    
    data['volatility_momentum'] = (data['close'] - data['close'].shift(1)) * \
                                 safe_divide(hl_range, data['high'].shift(1) - data['low'].shift(1)) * \
                                 safe_divide(data['volume'], data['volume'].shift(1)) * \
                                 np.sign(data['close'] - data['open'])
    
    data['volume_breakout_pressure'] = safe_divide(data['volume'], data['volume'].shift(1)) * \
                                      safe_divide(data['close'] - (data['high'] + data['low'])/2, hl_range) * \
                                      np.abs(data['open'] - data['close'].shift(1)) / hl_range
    
    # Volatility Position Analysis
    data['vol_range_efficiency'] = safe_divide(data['close'] - data['low'], hl_range) * \
                                  safe_divide(hl_range, data['high'].shift(1) - data['low'].shift(1)) * \
                                  np.sign(data['close'] - data['close'].shift(1))
    
    data['position_vol_fractal'] = data['vol_range_efficiency'] * \
                                  safe_divide(data['volume'], data['amount']) * \
                                  np.abs(data['close'] - data['close'].shift(3)) / hl_range
    
    # Volatility Entropy Dynamics
    price_position = safe_divide(data['close'] - data['low'], hl_range)
    price_position = np.clip(price_position, 1e-10, 1-1e-10)  # Avoid log(0)
    data['price_vol_entropy'] = -price_position * np.log(price_position) * \
                               safe_divide(hl_range, data['close'].shift(1)) * \
                               np.sign(data['close'] - data['open'])
    
    vol_change = safe_divide(data['volume'] - data['volume'].shift(1), data['volume'].shift(1))
    vol_change_abs = np.abs(vol_change)
    vol_change_abs = np.clip(vol_change_abs, 1e-10, 1-1e-10)  # Avoid log(0)
    data['volume_vol_entropy'] = -vol_change * np.log(vol_change_abs) * \
                                safe_divide(data['volume'], data['amount']) * \
                                np.sign(data['close'] - data['open'])
    
    # Volatility Regime Classification
    high_vol_expansion = (data['intraday_vol_regime'] > 0.02) & \
                        (safe_divide(data['volume'], data['volume'].shift(1)) > 1.5) & \
                        (data['volatility_momentum'] > 0.01)
    
    medium_vol_trend = (data['short_term_vol_persistence'] > 0.015) & \
                      (data['vol_vol_divergence'] > 0.2) & \
                      (data['vol_weighted_price_fractal'] > 0.3)
    
    low_vol_contraction = (data['medium_term_vol_expansion'] < 0.8) & \
                         (data['vol_vol_divergence'] < -0.2) & \
                         (data['vol_range_efficiency'] < 0.4)
    
    # Hierarchical Volatility-Volume Alpha Assembly
    vol_regime_component = (data['intraday_vol_regime'] * 0.4 + 
                           data['short_term_vol_persistence'] * 0.35 + 
                           data['medium_term_vol_expansion'] * 0.25)
    
    vol_asymmetry_component = (data['high_side_vol_pressure'] * 0.4 + 
                              data['low_side_vol_pressure'] * 0.4 + 
                              data['vol_vol_divergence'] * 0.2)
    
    price_vol_dynamics = (data['vol_weighted_price_fractal'] * 0.5 + 
                         data['volatility_momentum'] * 0.3 + 
                         data['volume_breakout_pressure'] * 0.2)
    
    position_context = vol_regime_component * data['position_vol_fractal'] * 0.35
    
    entropy_adjustment = (data['price_vol_entropy'] + data['volume_vol_entropy']) * 0.25
    
    # Regime multiplier
    regime_multiplier = np.where(high_vol_expansion, 1.5,
                                np.where(medium_vol_trend, 1.1,
                                        np.where(low_vol_contraction, 0.7, 1.0)))
    
    # Final alpha
    final_alpha = (position_context * vol_asymmetry_component * price_vol_dynamics + 
                  entropy_adjustment) * regime_multiplier
    
    return final_alpha
