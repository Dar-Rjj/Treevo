import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # 5-Day Momentum Components
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
        
        # 20-Day Momentum Components
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        divergence_20 = price_momentum_20 - volume_momentum_20
        
        # Volatility Assessment
        vol_5 = returns.iloc[i-4:i+1].std() if i >= 4 else 0
        vol_20 = returns.iloc[i-19:i+1].std()
        vol_ratio = vol_5 / vol_20 if vol_20 != 0 else 1
        
        # Dynamic Timeframe Weighting
        if vol_ratio > 1.5:
            weight_5, weight_20 = 0.7, 0.3
        elif vol_ratio >= 0.67:
            weight_5, weight_20 = 0.5, 0.5
        else:
            weight_5, weight_20 = 0.3, 0.7
        
        # Blended Divergence Score
        blended_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        
        # Volume Spike Confirmation
        volume_ma_20 = current_data['volume'].iloc[i-19:i+1].mean()
        volume_std_20 = current_data['volume'].iloc[i-19:i+1].std()
        current_volume = current_data['volume'].iloc[i]
        
        if current_volume > (volume_ma_20 + 2 * volume_std_20):
            volume_multiplier = 2.0
        elif current_volume > (volume_ma_20 + 1.5 * volume_std_20):
            volume_multiplier = 1.5
        else:
            volume_multiplier = 1.0
        
        # Volume-Enhanced Score
        volume_enhanced_score = blended_divergence * volume_multiplier
        
        # Key Price Level Proximity
        high_20 = current_data['high'].iloc[i-19:i+1].max()
        low_20 = current_data['low'].iloc[i-19:i+1].min()
        current_close = current_data['close'].iloc[i]
        
        if high_20 != low_20:
            price_position = (current_close - low_20) / (high_20 - low_20)
        else:
            price_position = 0.5
        
        if price_position > 0.8:
            adjustment_factor = 0.8
        elif price_position < 0.2:
            adjustment_factor = 1.2
        else:
            adjustment_factor = 1.0
        
        # Final Alpha Construction
        final_alpha = volume_enhanced_score * adjustment_factor
        result.iloc[i] = final_alpha
    
    return result
