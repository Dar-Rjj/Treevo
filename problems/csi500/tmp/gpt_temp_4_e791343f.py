import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Momentum-Volume Divergence
        # 5-Day Period
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
        
        # 20-Day Period
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        divergence_20 = price_momentum_20 - volume_momentum_20
        
        # Volatility Regime Weighting
        # Volatility Assessment
        vol_5 = returns.iloc[i-4:i+1].std() if i >= 4 else returns.iloc[:i+1].std()
        vol_20 = returns.iloc[i-19:i+1].std()
        volatility_ratio = vol_5 / vol_20 if vol_20 != 0 else 1.0
        
        # Dynamic Weight Assignment
        if volatility_ratio > 1.5:
            weight_5 = 0.8
            weight_20 = 0.2
        elif volatility_ratio >= 0.67:
            weight_5 = 0.5
            weight_20 = 0.5
        else:
            weight_5 = 0.2
            weight_20 = 0.8
        
        # Volume Spike Confirmation
        volume_20_avg = current_data['volume'].iloc[i-19:i+1].mean()
        volume_20_std = current_data['volume'].iloc[i-19:i+1].std()
        
        double_volume_spike = current_data['volume'].iloc[i] > (2 * volume_20_avg)
        spike_multiplier = 2.0 if double_volume_spike else 1.0
        
        # Key Price Level Proximity
        high_20 = current_data['high'].iloc[i-19:i+1].max()
        low_20 = current_data['low'].iloc[i-19:i+1].min()
        
        distance_to_high = (high_20 - current_data['close'].iloc[i]) / current_data['close'].iloc[i]
        distance_to_low = (current_data['close'].iloc[i] - low_20) / current_data['close'].iloc[i]
        proximity_factor = min(distance_to_high, distance_to_low)
        
        # Dynamic Threshold Adjustment
        adaptive_threshold = 0.1 * vol_5
        strong_signal = 1.5 if proximity_factor > adaptive_threshold else 1.0
        weak_signal = 0.7 if proximity_factor <= adaptive_threshold else 1.0
        signal_strength = strong_signal if proximity_factor > adaptive_threshold else weak_signal
        
        # Final Alpha Construction
        blended_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        volume_enhanced_signal = blended_divergence * spike_multiplier
        proximity_adjusted_signal = volume_enhanced_signal * signal_strength
        final_alpha = proximity_adjusted_signal * (1 + proximity_factor)
        
        alpha_factor.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
