import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Multi-Scale Pressure Building
        # Micro-scale pressure
        if i >= 1:
            micro_pressure = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) * 
                            current_data['volume'].iloc[i] / 
                            (current_data['high'].iloc[i] - current_data['low'].iloc[i]))
        else:
            micro_pressure = 0
        
        # Meso-scale pressure
        if i >= 3:
            meso_vol_avg = current_data['volume'].iloc[i-2:i+1].mean()
            meso_high_max = current_data['high'].iloc[i-2:i+1].max()
            meso_low_min = current_data['low'].iloc[i-2:i+1].min()
            meso_pressure = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) * 
                           meso_vol_avg / (meso_high_max - meso_low_min))
        else:
            meso_pressure = 0
        
        # Macro-scale pressure
        if i >= 5:
            macro_vol_avg = current_data['volume'].iloc[i-4:i+1].mean()
            macro_high_max = current_data['high'].iloc[i-4:i+1].max()
            macro_low_min = current_data['low'].iloc[i-4:i+1].min()
            macro_pressure = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-5]) * 
                            macro_vol_avg / (macro_high_max - macro_low_min))
        else:
            macro_pressure = 0
        
        # Pressure Fractal Analysis
        if meso_pressure != 0:
            micro_meso_ratio = micro_pressure / meso_pressure if meso_pressure != 0 else 0
        else:
            micro_meso_ratio = 0
            
        if macro_pressure != 0:
            meso_macro_ratio = meso_pressure / macro_pressure if macro_pressure != 0 else 0
        else:
            meso_macro_ratio = 0
            
        pressure_coherence = (micro_pressure + meso_pressure + macro_pressure) / 3
        
        # Volume-Pressure Efficiency Dynamics
        # Micro efficiency
        if i >= 1:
            micro_efficiency = (abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / 
                              (current_data['volume'].iloc[i] * (current_data['high'].iloc[i] - current_data['low'].iloc[i])))
        else:
            micro_efficiency = 0
        
        # Meso efficiency
        if i >= 3:
            meso_efficiency = (abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / 
                             (meso_vol_avg * (meso_high_max - meso_low_min)))
        else:
            meso_efficiency = 0
        
        # Macro efficiency
        if i >= 5:
            macro_efficiency = (abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-5]) / 
                              (macro_vol_avg * (macro_high_max - macro_low_min)))
        else:
            macro_efficiency = 0
        
        # Efficiency-Pressure Alignment
        micro_alignment = micro_pressure * micro_efficiency
        meso_alignment = meso_pressure * meso_efficiency
        macro_alignment = macro_pressure * macro_efficiency
        
        # Fractal Breakout Pressure Signals
        if i >= 6:
            # Range compression
            range_avg = (current_data['high'].iloc[i-5:i] - current_data['low'].iloc[i-5:i]).mean()
            range_compression = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / range_avg if range_avg != 0 else 1
            
            # Volume compression
            vol_avg = current_data['volume'].iloc[i-5:i].mean()
            vol_compression = current_data['volume'].iloc[i] / vol_avg if vol_avg != 0 else 1
            
            # Pressure compression
            macro_pressure_avg = abs(current_data['close'].iloc[i-5:i] - current_data['close'].iloc[i-10:i-5]).mean()
            pressure_compression = abs(macro_pressure) / macro_pressure_avg if macro_pressure_avg != 0 else 1
        else:
            range_compression = vol_compression = pressure_compression = 1
        
        # Multi-Scale Breakout Signals
        micro_breakout = range_compression < 0.7 and micro_pressure > 0
        meso_breakout = (range_compression < 0.8 and vol_compression < 0.8 and 
                        pressure_compression < 0.8 and meso_pressure > 0)
        macro_breakout = (range_compression < 0.9 and vol_compression < 0.9 and 
                         pressure_compression < 0.9 and macro_pressure > 0)
        
        # Pressure Flow Momentum
        if i >= 1:
            current_amount = current_data['amount'].iloc[i]
            if current_amount > 0:
                if current_data['close'].iloc[i] > current_data['open'].iloc[i]:
                    buying_pressure = current_amount
                    selling_pressure = 0
                else:
                    buying_pressure = 0
                    selling_pressure = current_amount
                net_pressure_flow = (buying_pressure - selling_pressure) / current_amount
            else:
                net_pressure_flow = 0
        else:
            net_pressure_flow = 0
        
        # Multi-Timeframe Pressure Alignment
        micro_meso_alignment = micro_pressure * meso_pressure
        meso_macro_alignment = meso_pressure * macro_pressure
        cross_scale_alignment = micro_pressure * meso_pressure * macro_pressure
        
        # Regime Identification
        if micro_meso_ratio > 1.1 and meso_macro_ratio > 1.1:
            regime = 'accumulation'
        elif micro_meso_ratio < 0.9 and meso_macro_ratio < 0.9:
            regime = 'distribution'
        else:
            regime = 'transition'
        
        # Adaptive Pressure Alpha Synthesis
        # Base alpha calculation
        base_alpha = (micro_pressure + meso_pressure + macro_pressure) / 3
        
        # Regime-based weighting
        if regime == 'accumulation':
            weighted_pressure = (0.2 * micro_pressure + 0.2 * meso_pressure + 0.6 * macro_pressure)
        elif regime == 'distribution':
            weighted_pressure = (0.7 * micro_pressure + 0.2 * meso_pressure + 0.1 * macro_pressure)
        else:  # transition
            weighted_pressure = base_alpha
        
        # Breakout enhancement
        if macro_breakout:
            weighted_pressure *= 1.3
        elif meso_breakout:
            weighted_pressure *= 1.2
        elif micro_breakout:
            weighted_pressure *= 1.1
        
        # Efficiency refinement
        efficiency_avg = (micro_efficiency + meso_efficiency + macro_efficiency) / 3
        if efficiency_avg > 0:
            weighted_pressure *= efficiency_avg
        
        # Alignment adjustment
        if cross_scale_alignment > 0:
            weighted_pressure *= (1 + 0.2 * cross_scale_alignment)
        
        # Flow divergence adjustment
        if (net_pressure_flow > 0 and base_alpha < 0) or (net_pressure_flow < 0 and base_alpha > 0):
            weighted_pressure *= -1
        
        # Temporal smoothing based on pressure coherence
        if i >= 3:
            recent_pressure = [micro_pressure, meso_pressure, macro_pressure]
            pressure_variance = np.var(recent_pressure)
            
            if pressure_variance < 0.1:  # High coherence
                if i >= 5:
                    alpha_3day_avg = alpha.iloc[i-2:i+1].mean()
                    final_alpha = 0.7 * weighted_pressure + 0.3 * alpha_3day_avg
                else:
                    final_alpha = weighted_pressure
            elif pressure_variance > 0.5:  # Low coherence
                if i >= 1:
                    final_alpha = 0.8 * weighted_pressure + 0.2 * alpha.iloc[i-1]
                else:
                    final_alpha = weighted_pressure
            else:  # Medium coherence
                final_alpha = weighted_pressure
        else:
            final_alpha = weighted_pressure
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial values
    alpha = alpha.fillna(0)
    
    return alpha
