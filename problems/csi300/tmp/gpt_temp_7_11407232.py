import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required rolling windows
    for i in range(len(df)):
        if i < 13:  # Need at least 13 days of data
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1] if i > 0 else None
        
        # Multi-Scale Volume-Price Fractal Framework
        # Volume-Price Fractal Asymmetry
        if prev_data is not None:
            micro_fractal = ((current_data['close'] - current_data['low']) / 
                           (current_data['high'] - current_data['low'] + 1e-8) * 
                           (current_data['volume'] / (prev_data['volume'] + 1e-8)))
        else:
            micro_fractal = 0
            
        # Meso fractal (5-day window)
        meso_window = df.iloc[i-4:i+1]  # t-4 to t
        meso_fractal = ((current_data['close'] - meso_window['low'].min()) / 
                       (meso_window['high'].max() - meso_window['low'].min() + 1e-8) * 
                       (current_data['volume'] / (meso_window.iloc[:-1]['volume'].mean() + 1e-8)))
        
        # Macro fractal (13-day window)
        macro_window = df.iloc[i-12:i+1]  # t-12 to t
        macro_fractal = ((current_data['close'] - macro_window['low'].min()) / 
                        (macro_window['high'].max() - macro_window['low'].min() + 1e-8) * 
                        (current_data['volume'] / (macro_window.iloc[:-1]['volume'].mean() + 1e-8)))
        
        volume_price_fractal_cascade = micro_fractal * meso_fractal * macro_fractal
        
        # Dynamic Volume-Price Momentum
        if prev_data is not None:
            volume_weighted_price_momentum = ((current_data['close'] - prev_data['close']) * 
                                            current_data['volume'] / (current_data['high'] - current_data['low'] + 1e-8))
        else:
            volume_weighted_price_momentum = 0
            
        volume_weighted_range_momentum = ((current_data['close'] - meso_window['low'].min()) * 
                                        current_data['volume'] / (meso_window['high'].max() - meso_window['low'].min() + 1e-8))
        
        volume_weighted_trend_momentum = ((current_data['close'] - macro_window['low'].min()) * 
                                        current_data['volume'] / (macro_window['high'].max() - macro_window['low'].min() + 1e-8))
        
        # Fractal Volume-Price Reversal Patterns
        if prev_data is not None:
            micro_reversal = ((current_data['close'] - prev_data['close']) / 
                            (current_data['high'] - current_data['low'] + 1e-8) * 
                            (current_data['volume'] / (prev_data['volume'] + 1e-8)))
        else:
            micro_reversal = 0
            
        meso_reversal = ((current_data['close'] - df.iloc[i-5]['close']) / 
                        (meso_window['high'].max() - meso_window['low'].min() + 1e-8) * 
                        (current_data['volume'] / (meso_window.iloc[:-1]['volume'].mean() + 1e-8)))
        
        macro_reversal = ((current_data['close'] - df.iloc[i-13]['close']) / 
                         (macro_window['high'].max() - macro_window['low'].min() + 1e-8) * 
                         (current_data['volume'] / (macro_window.iloc[:-1]['volume'].mean() + 1e-8)))
        
        # Dynamic Range-Volume Divergence Analysis
        # Adaptive Range Framework
        if prev_data is not None:
            dynamic_true_range = (max(current_data['high'] - current_data['low'], 
                                   abs(current_data['high'] - prev_data['close']), 
                                   abs(current_data['low'] - prev_data['close'])) * 
                               (current_data['volume'] / (prev_data['volume'] + 1e-8)))
        else:
            dynamic_true_range = 0
            
        # 5-day Volume-Weighted Range
        volume_weighted_range_5 = ((meso_window['high'] - meso_window['low']) * meso_window['volume']).sum() / (meso_window['volume'].sum() + 1e-8)
        
        # 13-day Volume-Weighted Range  
        volume_weighted_range_13 = ((macro_window['high'] - macro_window['low']) * macro_window['volume']).sum() / (macro_window['volume'].sum() + 1e-8)
        
        # Volume-Price Divergence Patterns
        if prev_data is not None:
            micro_divergence = (((current_data['close'] / (prev_data['close'] + 1e-8)) - 1) * 
                              ((current_data['volume'] / (prev_data['volume'] + 1e-8)) - 1) * 
                              (current_data['high'] - current_data['low']))
        else:
            micro_divergence = 0
            
        meso_divergence = (((current_data['close'] / (df.iloc[i-5]['close'] + 1e-8)) - 1) * 
                          ((current_data['volume'] / (meso_window.iloc[:-1]['volume'].mean() + 1e-8)) - 1) * 
                          (meso_window['high'].max() - meso_window['low'].min()))
        
        macro_divergence = (((current_data['close'] / (df.iloc[i-13]['close'] + 1e-8)) - 1) * 
                           ((current_data['volume'] / (macro_window.iloc[:-1]['volume'].mean() + 1e-8)) - 1) * 
                           (macro_window['high'].max() - macro_window['low'].min()))
        
        fractal_divergence_cascade = micro_divergence * meso_divergence * macro_divergence
        
        # Dynamic Divergence Integration
        short_term_divergence_weight = fractal_divergence_cascade / (volume_weighted_range_5 + 0.001)
        long_term_divergence_weight = fractal_divergence_cascade / (volume_weighted_range_13 + 0.001)
        dynamic_divergence_persistence = short_term_divergence_weight * long_term_divergence_weight
        
        # Fractal Volume-Efficiency Analysis
        # Volume-Weighted Movement Efficiency
        micro_efficiency = ((current_data['close'] - current_data['open']) / 
                          (current_data['high'] - current_data['low'] + 1e-8) * 
                          (current_data['volume'] / (prev_data['volume'] + 1e-8)) if prev_data is not None else 0)
        
        meso_efficiency = ((current_data['close'] - df.iloc[i-5]['close']) / 
                         ((meso_window['high'] - meso_window['low']).sum() + 1e-8) * 
                         (current_data['volume'] / (meso_window.iloc[:-1]['volume'].mean() + 1e-8)))
        
        macro_efficiency = ((current_data['close'] - df.iloc[i-13]['close']) / 
                          ((macro_window['high'] - macro_window['low']).sum() + 1e-8) * 
                          (current_data['volume'] / (macro_window.iloc[:-1]['volume'].mean() + 1e-8)))
        
        fractal_efficiency_ratio = micro_efficiency * meso_efficiency * macro_efficiency
        
        # Dynamic Volume Confirmation
        volume_range_acceleration = ((current_data['volume'] / (prev_data['volume'] + 1e-8)) / 
                                   (current_data['high'] - current_data['low'] + 1e-8) if prev_data is not None else 0)
        
        volume_range_breakout = ((current_data['volume'] / (meso_window.iloc[:-1]['volume'].max() + 1e-8)) * 
                               (current_data['high'] - current_data['low']) / 
                               (meso_window.iloc[:-1]['high'] - meso_window.iloc[:-1]['low']).mean())
        
        # Count volume-range persistence (last 3 days)
        volume_range_persistence = 0
        if i >= 3:
            for j in range(1, 4):
                if (df.iloc[i-j+1]['volume'] > df.iloc[i-j]['volume'] and 
                    (df.iloc[i-j+1]['high'] - df.iloc[i-j+1]['low']) > (df.iloc[i-j]['high'] - df.iloc[i-j]['low'])):
                    volume_range_persistence += 1
        
        dynamic_volume_range_composite = volume_range_acceleration * volume_range_breakout * (volume_range_persistence + 1)
        
        # Volume-Efficiency Synchronization
        volume_price_alignment = np.sign(fractal_efficiency_ratio) * np.sign(dynamic_volume_range_composite)
        volume_efficiency_divergence = abs(fractal_efficiency_ratio - dynamic_volume_range_composite)
        synchronized_volume_efficiency = fractal_efficiency_ratio * dynamic_volume_range_composite * (1 - volume_efficiency_divergence)
        
        # Multi-Timeframe Volume-Price Integration
        # Core Volume-Price Momentum Component
        base_volume_price_momentum = volume_price_fractal_cascade * volume_weighted_price_momentum
        divergence_enhancement = base_volume_price_momentum * fractal_divergence_cascade
        efficiency_confirmation = divergence_enhancement * synchronized_volume_efficiency
        
        # Dynamic Range-Volume Framework
        micro_range_volume_weight = efficiency_confirmation / (dynamic_true_range + 0.001)
        meso_range_volume_weight = efficiency_confirmation / (volume_weighted_range_5 + 0.001)
        macro_range_volume_weight = efficiency_confirmation / (volume_weighted_range_13 + 0.001)
        dynamic_range_volume_cascade = micro_range_volume_weight * meso_range_volume_weight * macro_range_volume_weight
        
        # Timeframe Volume-Price Synchronization
        short_term_convergence = micro_reversal * volume_range_acceleration
        medium_term_confirmation = meso_reversal * volume_range_breakout
        long_term_validation = macro_reversal * (volume_range_persistence + 1)
        multi_timeframe_alignment = short_term_convergence * medium_term_confirmation * long_term_validation
        
        # Final Composite Factor
        dynamic_volume_price_core = dynamic_range_volume_cascade * dynamic_divergence_persistence
        volume_price_momentum_integration = dynamic_volume_price_core * multi_timeframe_alignment
        final_alpha_factor = volume_price_momentum_integration * efficiency_confirmation
        
        result.iloc[i] = final_alpha_factor
    
    return result
