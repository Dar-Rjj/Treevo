import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Fractal Momentum Structure
        # Multi-Timeframe Momentum Fractals
        intraday_fractal = ((current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / 
                           (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8) * 
                           (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-2]))
        
        short_term_fractal = ((current_data['close'].iloc[-1] - current_data['close'].iloc[-4]) / 
                             (current_data['high'].iloc[-4:-1].max() - current_data['low'].iloc[-4:-1].min() + 1e-8) * 
                             (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-4]))
        
        medium_term_fractal = ((current_data['close'].iloc[-1] - current_data['close'].iloc[-11]) / 
                              (current_data['high'].iloc[-11:-1].max() - current_data['low'].iloc[-11:-1].min() + 1e-8) * 
                              (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-11]))
        
        # Fractal Momentum Divergence
        intra_short_divergence = intraday_fractal - short_term_fractal
        short_medium_divergence = short_term_fractal - medium_term_fractal
        fractal_divergence_product = intra_short_divergence * short_medium_divergence
        
        # Price-Range Fractal Enhancement
        range_efficiency = ((current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / 
                           (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8))
        range_expansion = ((current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / 
                          (current_data['high'].iloc[-2] - current_data['low'].iloc[-2] + 1e-8))
        range_fractal_momentum = (range_efficiency * range_expansion * 
                                 (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-3]))
        
        # Liquidity Fracture Patterns
        # Volume Concentration Fractures
        volume_spike_fracture = ((current_data['volume'].iloc[-1] / current_data['volume'].iloc[-6]) * 
                                ((current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / 
                                 (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)))
        volume_persistence = ((current_data['volume'].iloc[-1] / current_data['volume'].iloc[-2]) * 
                             (current_data['volume'].iloc[-2] / current_data['volume'].iloc[-3]))
        volume_concentration = (volume_spike_fracture * volume_persistence * 
                               np.sign(intraday_fractal))
        
        # Amount-Based Fractures
        amount_efficiency = (current_data['amount'].iloc[-1] / 
                            (current_data['volume'].iloc[-1] * current_data['close'].iloc[-1] + 1e-8))
        amount_momentum = ((current_data['amount'].iloc[-1] / current_data['amount'].iloc[-2]) * 
                          ((current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / 
                           (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)))
        amount_fracture_alignment = (amount_efficiency * amount_momentum * 
                                    (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-4]))
        
        # Regime-Dependent Fracture Dynamics
        # Volatility Regime Detection
        vol_ratio_current = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / current_data['close'].iloc[-2]
        vol_ratio_short = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / (current_data['high'].iloc[-6] - current_data['low'].iloc[-6] + 1e-8)
        vol_ratio_long = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / (current_data['high'].iloc[-11] - current_data['low'].iloc[-11] + 1e-8)
        
        high_volatility = vol_ratio_current > 0.035 and vol_ratio_short > 1.3
        low_volatility = vol_ratio_current < 0.012 and vol_ratio_long < 0.75
        
        # Volume Regime Detection
        volume_ratio_short = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-4]
        volume_ratio_very_short = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-2]
        
        high_volume = volume_ratio_short > 1.5 and volume_ratio_very_short > 1.2
        low_volume = volume_ratio_short < 0.7 and volume_ratio_very_short < 0.9
        
        # Regime-Adaptive Fracture Weighting
        if high_volatility:
            regime_fractal = intraday_fractal * 0.8 + short_term_fractal * 0.2
        elif low_volatility:
            regime_fractal = short_term_fractal * 0.6 + medium_term_fractal * 0.4
        else:
            regime_fractal = (intraday_fractal + short_term_fractal + medium_term_fractal) / 3
        
        if high_volume:
            volume_concentration *= 1.4
        if low_volume:
            amount_fracture_alignment *= 0.6
        
        # Fracture Convergence Mechanics
        # Multi-Fractal Alignment
        timeframe_alignment = (intra_short_divergence * short_medium_divergence * 
                              (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-3]))
        liquidity_price_alignment = (volume_concentration * amount_fracture_alignment * 
                                    np.sign(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]))
        fractal_convergence_signal = timeframe_alignment * liquidity_price_alignment
        
        # Fracture Stability Assessment
        volatility_stability = ((current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / 
                               (current_data['high'].iloc[-6] - current_data['low'].iloc[-6] + 1e-8))
        volume_stability = (current_data['volume'].iloc[-1] / 
                           ((current_data['volume'].iloc[-2] + current_data['volume'].iloc[-3] + 
                             current_data['volume'].iloc[-4]) / 3 + 1e-8))
        fracture_stability = volatility_stability * volume_stability * range_expansion
        
        # Alpha Synthesis Framework
        # Core Fracture Momentum
        base_fracture_signal = fractal_divergence_product * volume_concentration
        enhanced_fracture = base_fracture_signal * amount_fracture_alignment
        regime_enhanced_fracture = enhanced_fracture * regime_fractal
        
        # Convergence Integration
        alignment_boost = regime_enhanced_fracture * (1 + fractal_convergence_signal * 0.3)
        stability_adjustment = alignment_boost * fracture_stability
        range_confirmation = stability_adjustment * range_fractal_momentum
        
        # Final Alpha Construction
        primary_alpha = range_confirmation * (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-6])
        secondary_enhancement = primary_alpha * (1 + amount_efficiency * 0.2)
        final_output = secondary_enhancement * np.sign(fractal_divergence_product)
        
        alpha.iloc[i] = final_output
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
