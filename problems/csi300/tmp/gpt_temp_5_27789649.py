import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Volatility Context
    data['short_term_vol'] = data['high'].shift(4) - data['low'].shift(4)
    data['medium_term_vol'] = data['high'].shift(9) - data['low'].shift(9)
    data['vol_ratio'] = (data['high'] - data['low']) / data['medium_term_vol']
    
    # Microstructure Efficiency Patterns
    data['price_range_eff'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['upper_shadow'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['lower_shadow'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['net_rejection'] = data['lower_shadow'] - data['upper_shadow']
    data['position_eff'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Volume-Liquidity Dynamics
    data['daily_accel'] = data['volume'] / data['volume'].shift(1)
    data['medium_term_momentum'] = data['volume'] / data['volume'].shift(4)
    data['accel_score'] = data['daily_accel'] * data['medium_term_momentum']
    
    data['current_intensity'] = data['amount'] / data['volume']
    data['intensity_momentum'] = data['current_intensity'] / (data['amount'].shift(4) / data['volume'].shift(4))
    
    # Size Imbalance - rolling calculation
    intensity_rolling = []
    for i in range(len(data)):
        if i >= 10:
            window_intensity = (data['amount'].iloc[i-10:i] / data['volume'].iloc[i-10:i]).mean()
            current_int = data['current_intensity'].iloc[i]
            size_imbalance = window_intensity / (10 * current_int) if current_int != 0 else 0
        else:
            size_imbalance = 0
        intensity_rolling.append(size_imbalance)
    data['size_imbalance'] = intensity_rolling
    
    # Volume Breakout - rolling calculation
    volume_rolling = []
    for i in range(len(data)):
        if i >= 20:
            max_vol = data['volume'].iloc[i-20:i].max()
            volume_breakout = data['volume'].iloc[i] / max_vol if max_vol != 0 else 0
        else:
            volume_breakout = 0
        volume_rolling.append(volume_breakout)
    data['volume_breakout'] = volume_rolling
    
    data['vol_momentum'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    data['co_movement_score'] = data['volume'] * (data['high'] - data['low'])
    
    # Price-Volume-Amount Integration
    up_flow = np.where(data['close'] > data['open'], (data['close'] - data['open']) * data['volume'], 0)
    down_flow = np.where(data['close'] < data['open'], (data['open'] - data['close']) * data['volume'], 0)
    data['net_flow'] = up_flow - down_flow
    
    bullish_pressure = np.where(data['close'] > data['open'], (data['close'] - data['open']) * data['current_intensity'], 0)
    bearish_pressure = np.where(data['close'] < data['open'], (data['open'] - data['close']) * data['current_intensity'], 0)
    data['net_pressure'] = bullish_pressure - bearish_pressure
    
    data['effective_spread'] = np.abs(data['close'] - (data['high'] + data['low'])/2) / ((data['high'] - data['low'])/2)
    data['volume_weighted_spread'] = data['effective_spread'] * data['volume']
    data['flow_direction'] = data['volume_weighted_spread'] * (data['close'] - data['open']) * data['volume'] / data['amount']
    
    # Multi-Dimensional Composite Factors
    data['efficiency_vol_score'] = data['price_range_eff'] * data['vol_ratio']
    data['rejection_vol'] = data['net_rejection'] * data['vol_momentum']
    data['position_vol'] = data['position_eff'] * data['short_term_vol']
    
    data['acceleration_pressure'] = data['accel_score'] * data['net_pressure']
    data['breakout_flow'] = data['volume_breakout'] * data['net_flow']
    data['intensity_flow'] = data['intensity_momentum'] * data['flow_direction']
    
    data['efficiency_liquidity'] = data['price_range_eff'] * data['accel_score']
    data['rejection_liquidity'] = data['net_rejection'] * data['size_imbalance']
    data['position_liquidity'] = data['position_eff'] * data['current_intensity']
    
    # Component Construction
    data['volatility_component'] = data['efficiency_vol_score'] * data['rejection_vol'] * data['position_vol']
    data['volume_component'] = data['acceleration_pressure'] * data['breakout_flow'] * data['intensity_flow']
    data['microstructure_component'] = data['efficiency_liquidity'] * data['rejection_liquidity'] * data['position_liquidity']
    data['flow_component'] = data['net_flow'] * data['net_pressure'] * data['flow_direction']
    
    # Regime-Based Integration
    high_vol_regime = data['vol_ratio'] > 1.2
    high_volume_regime = data['volume_breakout'] > 1.5
    
    # Initialize weights
    w_vol = np.zeros(len(data))
    w_volume = np.zeros(len(data))
    w_micro = np.zeros(len(data))
    w_flow = np.zeros(len(data))
    
    # Apply regime weights
    w_vol[high_vol_regime] = 0.4
    w_volume[high_vol_regime] = 0.3
    w_micro[high_vol_regime] = 0.2
    w_flow[high_vol_regime] = 0.1
    
    w_vol[high_volume_regime] = 0.2
    w_volume[high_volume_regime] = 0.5
    w_micro[high_volume_regime] = 0.2
    w_flow[high_volume_regime] = 0.1
    
    # Normal regime for remaining cases
    normal_regime = ~high_vol_regime & ~high_volume_regime
    w_vol[normal_regime] = 0.25
    w_volume[normal_regime] = 0.25
    w_micro[normal_regime] = 0.25
    w_flow[normal_regime] = 0.25
    
    # Final Alpha Generation
    weighted_sum = (data['volatility_component'] * w_vol + 
                   data['volume_component'] * w_volume + 
                   data['microstructure_component'] * w_micro + 
                   data['flow_component'] * w_flow)
    
    # Apply adjustments
    alpha = weighted_sum * data['accel_score']
    alpha = alpha / data['vol_ratio'].replace(0, 1)  # Avoid division by zero
    alpha = alpha + (data['position_eff'] * data['price_range_eff'])
    
    return alpha
