import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns for volatility
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Momentum-Volume Divergence Analysis
        # 5-Day Momentum Divergence
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
            
        # 20-Day Momentum Divergence
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        divergence_20 = price_momentum_20 - volume_momentum_20
        
        # 2. Dynamic Volatility Weighting
        # Volatility Calculation
        vol_5 = returns.iloc[i-4:i+1].std() if i >= 4 else 0.01
        vol_20 = returns.iloc[i-19:i+1].std()
        
        # Avoid division by zero
        if vol_20 == 0:
            vol_20 = 0.01
            
        volatility_ratio = vol_5 / vol_20
        
        # Adaptive Weight Assignment
        if volatility_ratio > 1.5:
            weight_5 = 0.8
            weight_20 = 0.2
        elif volatility_ratio >= 0.67:
            weight_5 = 0.5
            weight_20 = 0.5
        else:
            weight_5 = 0.2
            weight_20 = 0.8
        
        # 3. Statistical Volume Spike Confirmation
        # Volume Baseline
        volume_ma = current_data['volume'].iloc[i-19:i+1].mean()
        volume_std = current_data['volume'].iloc[i-19:i+1].std()
        
        current_volume = current_data['volume'].iloc[i]
        
        # Spike Classification
        if volume_std == 0:
            volume_multiplier = 1.0
        else:
            z_score = (current_volume - volume_ma) / volume_std
            if z_score > 2.5:
                volume_multiplier = 2.5
            elif z_score > 1.5:
                volume_multiplier = 1.8
            elif z_score > 0.8:
                volume_multiplier = 1.3
            else:
                volume_multiplier = 1.0
        
        # 4. Price-Level Proximity Analysis
        # Key Levels Identification
        high_20 = current_data['high'].iloc[i-19:i+1].max()
        low_20 = current_data['low'].iloc[i-19:i+1].min()
        current_close = current_data['close'].iloc[i]
        
        # Proximity-Based Adjustment
        if current_close > 0.9 * high_20:
            price_multiplier = 0.7
        elif current_close < 1.1 * low_20:
            price_multiplier = 1.3
        else:
            price_multiplier = 1.0
        
        # 5. Final Alpha Construction
        # Weighted Divergence Blend
        weighted_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        
        # Volume-Enhanced Score
        volume_enhanced_score = weighted_divergence * volume_multiplier
        
        # Context-Finalized Alpha
        alpha.iloc[i] = volume_enhanced_score * price_multiplier
    
    return alpha
