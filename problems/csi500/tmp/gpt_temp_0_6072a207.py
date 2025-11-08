import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # 5-Day Momentum Components
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            momentum_divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            momentum_divergence_5 = 0
        
        # 20-Day Momentum Components
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        momentum_divergence_20 = price_momentum_20 - volume_momentum_20
        
        # Dynamic Volatility Weighting
        if i >= 20:
            vol_5 = returns.iloc[i-4:i+1].std()
            vol_20 = returns.iloc[i-19:i+1].std()
            
            if vol_20 == 0:
                volatility_ratio = 1.0
            else:
                volatility_ratio = vol_5 / vol_20
            
            if volatility_ratio > 1.5:
                weight_5 = 0.8
                weight_20 = 0.2
            elif volatility_ratio >= 0.67:
                weight_5 = 0.5
                weight_20 = 0.5
            else:
                weight_5 = 0.2
                weight_20 = 0.8
        else:
            weight_5 = 0.5
            weight_20 = 0.5
        
        # Statistical Volume Spike Confirmation
        volume_window = current_data['volume'].iloc[i-19:i+1]
        volume_mean = volume_window.mean()
        volume_std = volume_window.std()
        current_volume = current_data['volume'].iloc[i]
        
        if volume_std == 0:
            volume_confidence = 1.0
        else:
            z_score = (current_volume - volume_mean) / volume_std
            
            if z_score > 2.5:
                volume_confidence = 2.0
            elif z_score > 1.5:
                volume_confidence = 1.5
            elif z_score > 0.5:
                volume_confidence = 1.2
            else:
                volume_confidence = 1.0
        
        # Price-Level Proximity Assessment
        high_20 = current_data['high'].iloc[i-19:i+1].max()
        low_20 = current_data['low'].iloc[i-19:i+1].min()
        current_close = current_data['close'].iloc[i]
        
        if current_close > 0.9 * high_20:
            price_adjustment = 0.7
        elif current_close < 1.1 * low_20:
            price_adjustment = 1.3
        else:
            price_adjustment = 1.0
        
        # Final Alpha Construction
        weighted_divergence = (momentum_divergence_5 * weight_5) + (momentum_divergence_20 * weight_20)
        volume_enhanced_signal = weighted_divergence * volume_confidence
        context_finalized_alpha = volume_enhanced_signal * price_adjustment
        
        alpha.iloc[i] = context_finalized_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
