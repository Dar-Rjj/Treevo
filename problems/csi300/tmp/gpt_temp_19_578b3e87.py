import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Regime Classification
    # Volatility-Flow Regime
    short_term_vol = []
    for i in range(len(data)):
        if i < 4:
            short_term_vol.append(np.nan)
            continue
        vol_sum = 0
        count = 0
        for j in range(i-4, i+1):
            if j > 0 and abs(data['close'].iloc[j] - data['close'].iloc[j-1]) > 0:
                vol_ratio = (data['high'].iloc[j] - data['low'].iloc[j]) / abs(data['close'].iloc[j] - data['close'].iloc[j-1])
                vol_sum += vol_ratio
                count += 1
        short_term_vol.append(vol_sum / count if count > 0 else np.nan)
    
    data['short_term_vol'] = short_term_vol
    
    medium_term_vol = []
    for i in range(len(data)):
        if i < 5:
            medium_term_vol.append(np.nan)
            continue
        if (data['high'].iloc[i-5] - data['low'].iloc[i-5]) > 0:
            vol_ratio = (data['high'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i-5] - data['low'].iloc[i-5])
            medium_term_vol.append(vol_ratio)
        else:
            medium_term_vol.append(np.nan)
    
    data['medium_term_vol'] = medium_term_vol
    
    volatility_flow_regime = []
    for i in range(len(data)):
        if i < 5 or pd.isna(data['short_term_vol'].iloc[i]) or pd.isna(data['medium_term_vol'].iloc[i]):
            volatility_flow_regime.append(np.nan)
            continue
        vol_diff_sign = np.sign(data['short_term_vol'].iloc[i] - data['medium_term_vol'].iloc[i])
        if i > 0 and data['volume'].iloc[i-1] > 0:
            volume_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1]
            volatility_flow_regime.append(vol_diff_sign * volume_ratio)
        else:
            volatility_flow_regime.append(np.nan)
    
    data['volatility_flow_regime'] = volatility_flow_regime
    
    # Flow-Persistence Regime
    flow_persistence_regime = []
    consecutive_count = 0
    for i in range(len(data)):
        if i == 0:
            flow_persistence_regime.append(np.nan)
            continue
        
        current_direction = np.sign(data['close'].iloc[i] - data['open'].iloc[i])
        prev_direction = np.sign(data['close'].iloc[i-1] - data['open'].iloc[i-1]) if i > 0 else 0
        
        if current_direction == prev_direction and current_direction != 0:
            consecutive_count += 1
        else:
            consecutive_count = 1 if current_direction != 0 else 0
        
        if i > 0 and data['volume'].iloc[i-1] > 0:
            volume_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1]
            flow_persistence_regime.append(consecutive_count * volume_ratio)
        else:
            flow_persistence_regime.append(np.nan)
    
    data['flow_persistence_regime'] = flow_persistence_regime
    
    # Multi-Scale Rejection-Flow Asymmetry
    # Intraday Flow-Rejection Asymmetry
    intraday_asymmetry = []
    for i in range(len(data)):
        if data['high'].iloc[i] - data['low'].iloc[i] > 0:
            upper_shadow = (data['high'].iloc[i] - max(data['open'].iloc[i], data['close'].iloc[i])) / (data['high'].iloc[i] - data['low'].iloc[i])
            lower_shadow = (min(data['open'].iloc[i], data['close'].iloc[i]) - data['low'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i])
            
            shadow_diff = upper_shadow - lower_shadow
            direction_sign = np.sign(data['close'].iloc[i] - data['open'].iloc[i])
            
            if i > 0 and data['volume'].iloc[i-1] > 0:
                volume_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1]
                intraday_asymmetry.append(shadow_diff * direction_sign * volume_ratio)
            else:
                intraday_asymmetry.append(np.nan)
        else:
            intraday_asymmetry.append(np.nan)
    
    data['intraday_asymmetry'] = intraday_asymmetry
    
    # Multi-Timeframe Flow-Rejection Momentum
    multi_timeframe_momentum = []
    for i in range(len(data)):
        if i < 2:
            multi_timeframe_momentum.append(np.nan)
            continue
            
        if data['high'].iloc[i] - data['low'].iloc[i] > 0:
            high_rejection = (data['high'].iloc[i] - max(data['close'].iloc[i-2], data['close'].iloc[i-1], data['close'].iloc[i])) / (data['high'].iloc[i] - data['low'].iloc[i])
            low_rejection = (min(data['close'].iloc[i-2], data['close'].iloc[i-1], data['close'].iloc[i]) - data['low'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i])
            
            rejection_diff = high_rejection - low_rejection
            direction_sign = np.sign(data['close'].iloc[i] - data['open'].iloc[i])
            
            if i > 0 and data['volume'].iloc[i-1] > 0:
                volume_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1]
                multi_timeframe_momentum.append(rejection_diff * direction_sign * volume_ratio)
            else:
                multi_timeframe_momentum.append(np.nan)
        else:
            multi_timeframe_momentum.append(np.nan)
    
    data['multi_timeframe_momentum'] = multi_timeframe_momentum
    
    # Flow-Adaptive Velocity Dynamics
    # Flow-Weighted Momentum
    flow_weighted_momentum = []
    for i in range(len(data)):
        if i == 0:
            flow_weighted_momentum.append(np.nan)
            continue
            
        if data['close'].iloc[i-1] > 0 and data['volume'].iloc[i-1] > 0:
            price_momentum = (data['close'].iloc[i] / data['close'].iloc[i-1] - 1)
            volume_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1]
            flow_weighted_momentum.append(price_momentum * volume_ratio)
        else:
            flow_weighted_momentum.append(np.nan)
    
    data['flow_weighted_momentum'] = flow_weighted_momentum
    
    # Range-Flow Efficiency Integration
    # Flow-Range Capture
    flow_range_capture = []
    for i in range(len(data)):
        if data['high'].iloc[i] - data['low'].iloc[i] > 0 and i > 0 and data['volume'].iloc[i-1] > 0:
            range_capture = abs(data['close'].iloc[i] - data['open'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i])
            volume_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1]
            flow_range_capture.append(range_capture * volume_ratio)
        else:
            flow_range_capture.append(np.nan)
    
    data['flow_range_capture'] = flow_range_capture
    
    # Composite Alpha Construction
    alpha_values = []
    for i in range(len(data)):
        if (pd.isna(data['volatility_flow_regime'].iloc[i]) or 
            pd.isna(data['flow_persistence_regime'].iloc[i]) or
            pd.isna(data['intraday_asymmetry'].iloc[i]) or
            pd.isna(data['multi_timeframe_momentum'].iloc[i]) or
            pd.isna(data['flow_weighted_momentum'].iloc[i]) or
            pd.isna(data['flow_range_capture'].iloc[i])):
            alpha_values.append(np.nan)
            continue
        
        core_signal = (data['intraday_asymmetry'].iloc[i] * 
                      data['multi_timeframe_momentum'].iloc[i] * 
                      data['flow_weighted_momentum'].iloc[i])
        
        regime_adjustment = (data['volatility_flow_regime'].iloc[i] * 
                           data['flow_persistence_regime'].iloc[i])
        
        final_alpha = regime_adjustment * core_signal * data['flow_range_capture'].iloc[i]
        alpha_values.append(final_alpha)
    
    return pd.Series(alpha_values, index=data.index)
