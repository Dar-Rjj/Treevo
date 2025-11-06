import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all entropy components
    data['range_efficiency_entropy'] = 0.0
    data['gap_volatility_entropy'] = 0.0
    data['volume_to_range_entropy'] = 0.0
    data['high_fractal_momentum_entropy'] = 0.0
    data['low_fractal_momentum_entropy'] = 0.0
    data['momentum_persistence_fractal'] = 0.0
    data['amount_efficiency_entropy'] = 0.0
    data['volume_breakout_entropy'] = 0.0
    data['efficiency_breakout_entropy'] = 0.0
    data['close_position_entropy'] = 0.0
    data['anchoring_fractal_position'] = 0.0
    
    # Calculate entropy components
    for i in range(5, len(data)):
        # Range Efficiency Entropy
        range_eff = (data['high'].iloc[i] - data['low'].iloc[i]) / (data['close'].iloc[i-1] + data['open'].iloc[i])
        range_eff *= 2 * abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        if range_eff > 0:
            data.loc[data.index[i], 'range_efficiency_entropy'] = -range_eff * np.log(range_eff)
        
        # Gap Volatility Entropy
        gap_vol = abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1])
        gap_vol *= data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1
        if gap_vol > 0:
            data.loc[data.index[i], 'gap_volatility_entropy'] = -gap_vol * np.log(gap_vol)
        
        # Volume-to-Range Entropy
        vol_range = data['volume'].iloc[i] / (data['high'].iloc[i] - data['low'].iloc[i])
        vol_range *= abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        if vol_range > 0:
            data.loc[data.index[i], 'volume_to_range_entropy'] = -vol_range * np.log(vol_range)
        
        # High Fractal Momentum Entropy
        high_fractal = (data['close'].iloc[i] - data['close'].iloc[i-2]) / (data['high'].iloc[i] - data['low'].iloc[i])
        high_fractal *= abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        if high_fractal > 0:
            data.loc[data.index[i], 'high_fractal_momentum_entropy'] = -high_fractal * np.log(high_fractal)
        
        # Low Fractal Momentum Entropy
        momentum_sum = sum(abs(data['close'].iloc[j] - data['close'].iloc[j-1]) for j in range(i-4, i+1))
        low_fractal = (data['close'].iloc[i] - data['close'].iloc[i-5]) / momentum_sum if momentum_sum > 0 else 0
        low_fractal *= data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1
        low_fractal *= abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i])
        if low_fractal > 0:
            data.loc[data.index[i], 'low_fractal_momentum_entropy'] = -low_fractal * np.log(low_fractal)
        
        # Momentum Persistence Fractal
        if (data['close'].iloc[i] - data['close'].iloc[i-1]) * (data['close'].iloc[i-1] - data['close'].iloc[i-2]) > 0:
            mom_pers = abs((data['close'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i]))
            if mom_pers > 0:
                data.loc[data.index[i], 'momentum_persistence_fractal'] = -mom_pers * np.log(mom_pers) * data['volume'].iloc[i]
        
        # Amount Efficiency Entropy
        amount_eff = data['amount'].iloc[i] / (data['volume'].iloc[i] * data['close'].iloc[i]) if data['volume'].iloc[i] * data['close'].iloc[i] > 0 else 0
        amount_eff *= data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1
        if amount_eff > 0:
            data.loc[data.index[i], 'amount_efficiency_entropy'] = -amount_eff * np.log(amount_eff)
        
        # Volume Breakout Entropy
        if (data['volume'].iloc[i] > 1.5 * data['volume'].iloc[i-1] and 
            (data['close'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i]) - 
            0.5 * abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i]) > 0.3):
            vol_break = abs((data['volume'].iloc[i] - data['volume'].iloc[i-1]) / data['volume'].iloc[i-1]) if data['volume'].iloc[i-1] > 0 else 0
            if vol_break > 0:
                data.loc[data.index[i], 'volume_breakout_entropy'] = -vol_break * np.log(vol_break)
        
        # Efficiency Breakout Entropy
        eff_break_val = data['amount'].iloc[i] / (data['volume'].iloc[i] * data['close'].iloc[i]) if data['volume'].iloc[i] * data['close'].iloc[i] > 0 else 0
        eff_break_val *= data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1
        if (eff_break_val > 1.5 and 
            abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1] > 0.6):
            amount_break = abs((data['amount'].iloc[i] - data['amount'].iloc[i-1]) / data['amount'].iloc[i-1]) if data['amount'].iloc[i-1] > 0 else 0
            if amount_break > 0:
                data.loc[data.index[i], 'efficiency_breakout_entropy'] = -amount_break * np.log(amount_break)
        
        # Close Position Entropy
        close_pos = (data['close'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i])
        close_pos -= 0.5 * abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i])
        if close_pos > 0:
            data.loc[data.index[i], 'close_position_entropy'] = -close_pos * np.log(close_pos)
        
        # Anchoring Fractal Position
        data.loc[data.index[i], 'anchoring_fractal_position'] = close_pos
    
    # Calculate final alpha components
    data['strong_entropic_breakout'] = ((data['high_fractal_momentum_entropy'] > 0.7) & 
                                       (data['volume_breakout_entropy'] > 0)).astype(float)
    
    data['moderate_entropic_trend'] = ((data['low_fractal_momentum_entropy'] > 0.5) & 
                                      (data['efficiency_breakout_entropy'] > 0)).astype(float)
    
    data['range_bound_entropy'] = ((abs(data['high_fractal_momentum_entropy']) < 0.2) & 
                                  (data['volume_to_range_entropy'].between(0.8, 1.2))).astype(float)
    
    data['liquidity_entropy_drain'] = ((data['volume_to_range_entropy'] > 2) & 
                                      (data['high_fractal_momentum_entropy'] < -0.4)).astype(float)
    
    data['entropic_exhaustion'] = ((data['amount_efficiency_entropy'] > 1.2) & 
                                  (data['volume_to_range_entropy'] < 0.5) & 
                                  (data['high_fractal_momentum_entropy'] < -0.7)).astype(float)
    
    # Volume Confirmation
    data['volume_confirmation'] = data['volume_breakout_entropy'] * data['volume_to_range_entropy']
    
    # Final Entropic Fractal Alpha
    data['entropic_fractal_alpha'] = (data['strong_entropic_breakout'] + 
                                     data['moderate_entropic_trend'] - 
                                     data['range_bound_entropy'] - 
                                     data['liquidity_entropy_drain'] - 
                                     data['entropic_exhaustion']) * data['volume_confirmation']
    
    return data['entropic_fractal_alpha']
