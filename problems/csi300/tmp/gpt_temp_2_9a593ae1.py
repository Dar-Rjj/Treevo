import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling windows
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # Fractal Gap Analysis
        if i >= 5:
            # Overnight Fractal Gap
            gap_magnitude = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1]
            
            high_5d = current_data['high'].iloc[i-5:i].max()
            low_5d = current_data['low'].iloc[i-5:i].min()
            fractal_context = (current_data['open'].iloc[i] - low_5d) / (high_5d - low_5d) if (high_5d - low_5d) > 0 else 0
            
            # Intraday Fractal Recovery
            recovery_strength = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / abs(current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) if abs(current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) > 0 else 0
            
            high_5d_current = current_data['high'].iloc[i-5:i+1].max()
            low_5d_current = current_data['low'].iloc[i-5:i+1].min()
            fractal_efficiency = (current_data['close'].iloc[i] - low_5d_current) / (high_5d_current - low_5d_current) if (high_5d_current - low_5d_current) > 0 else 0
            
            gap_momentum = gap_magnitude * fractal_context * recovery_strength * fractal_efficiency
        else:
            gap_momentum = 0
        
        # Asymmetric Microstructure
        if i >= 8:
            high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            upper_shadow = (current_data['high'].iloc[i] - current_data['close'].iloc[i]) / high_low_range if high_low_range > 0 else 0
            lower_shadow = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / high_low_range if high_low_range > 0 else 0
            net_rejection = lower_shadow - upper_shadow
            
            short_term_momentum = sum([(current_data['close'].iloc[j] - current_data['low'].iloc[j]) / (current_data['high'].iloc[j] - current_data['low'].iloc[j]) - 
                                      (current_data['high'].iloc[j] - current_data['close'].iloc[j]) / (current_data['high'].iloc[j] - current_data['low'].iloc[j]) 
                                      for j in range(i-3, i) if (current_data['high'].iloc[j] - current_data['low'].iloc[j]) > 0])
            
            medium_term_momentum = sum([(current_data['close'].iloc[j] - current_data['low'].iloc[j]) / (current_data['high'].iloc[j] - current_data['low'].iloc[j]) - 
                                       (current_data['high'].iloc[j] - current_data['close'].iloc[j]) / (current_data['high'].iloc[j] - current_data['low'].iloc[j]) 
                                       for j in range(i-8, i) if (current_data['high'].iloc[j] - current_data['low'].iloc[j]) > 0])
            
            enhanced_rejection = net_rejection * (current_data['volume'].iloc[i] / high_low_range if high_low_range > 0 else 0) * short_term_momentum * medium_term_momentum
        else:
            enhanced_rejection = 0
        
        # Asymmetric Liquidity
        if i >= 5:
            # Spread Asymmetry
            if current_data['close'].iloc[i] > current_data['open'].iloc[i]:
                bullish_pressure = (current_data['high'].iloc[i] - current_data['open'].iloc[i]) / current_data['close'].iloc[i] if current_data['close'].iloc[i] > 0 else 0
                bearish_pressure = 0
            elif current_data['close'].iloc[i] < current_data['open'].iloc[i]:
                bullish_pressure = 0
                bearish_pressure = (current_data['open'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i] if current_data['close'].iloc[i] > 0 else 0
            else:
                bullish_pressure = 0
                bearish_pressure = 0
            
            spread_asymmetry = bullish_pressure - bearish_pressure
            
            # Efficiency Asymmetry
            if current_data['close'].iloc[i] > current_data['open'].iloc[i]:
                bullish_efficiency = current_data['volume'].iloc[i] / high_low_range if high_low_range > 0 else 0
                bearish_efficiency = 0
            elif current_data['close'].iloc[i] < current_data['open'].iloc[i]:
                bullish_efficiency = 0
                bearish_efficiency = current_data['volume'].iloc[i] / high_low_range if high_low_range > 0 else 0
            else:
                bullish_efficiency = 0
                bearish_efficiency = 0
            
            efficiency_asymmetry = bullish_efficiency - bearish_efficiency
            
            # Fractal Order Flow
            upper_flow = abs(current_data['close'].iloc[i] - high_5d_current) / (high_low_range / 2) if (current_data['close'].iloc[i] < high_5d_current and high_low_range > 0) else 0
            lower_flow = abs(current_data['close'].iloc[i] - low_5d_current) / (high_low_range / 2) if (current_data['close'].iloc[i] > low_5d_current and high_low_range > 0) else 0
        else:
            spread_asymmetry = 0
            efficiency_asymmetry = 0
            upper_flow = 0
            lower_flow = 0
        
        # Regime Integration
        if i >= 10:
            volume_mean_10d = current_data['volume'].iloc[i-10:i].mean()
            
            if abs(gap_magnitude) > 0.02:
                # Gap-Dominated regime
                regime_alpha = gap_momentum * enhanced_rejection * (efficiency_asymmetry / (abs(spread_asymmetry) + 1e-8))
            elif current_data['volume'].iloc[i] > 2 * volume_mean_10d:
                # Breakout regime
                volume_max_20d = current_data['volume'].iloc[i-20:i].max() if i >= 20 else current_data['volume'].iloc[:i].max()
                breakout_alpha = (current_data['volume'].iloc[i] / volume_max_20d) * fractal_efficiency
                regime_alpha = breakout_alpha
            else:
                # Microstructure-Dominated regime
                regime_alpha = enhanced_rejection * (lower_flow - upper_flow) * (efficiency_asymmetry / (abs(spread_asymmetry) + 1e-8))
            
            # Adaptive Synthesis
            final_alpha = regime_alpha * (current_data['volume'].iloc[i] / (volume_mean_10d + 1e-8)) * (lower_flow - upper_flow)
        else:
            final_alpha = 0
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
