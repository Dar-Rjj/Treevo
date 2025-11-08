import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using momentum-volume divergence with dynamic volatility weighting,
    volume spike detection, and price-level proximity assessment.
    """
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Momentum-Volume Divergence Analysis
        # 5-Day Framework
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
        
        # 20-Day Framework
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        divergence_20 = price_momentum_20 - volume_momentum_20
        
        # 2. Dynamic Volatility Weighting
        # Volatility Assessment
        vol_5 = returns.iloc[i-4:i+1].std() if i >= 4 else 0
        vol_20 = returns.iloc[i-19:i+1].std()
        
        if vol_20 > 0:
            vol_ratio = vol_5 / vol_20
        else:
            vol_ratio = 1.0
        
        # Adaptive Weight Allocation
        if vol_ratio > 1.5:
            weight_5, weight_20 = 0.8, 0.2
        elif vol_ratio >= 0.67:
            weight_5, weight_20 = 0.5, 0.5
        else:
            weight_5, weight_20 = 0.2, 0.8
        
        # 3. Statistical Volume Spike Detection
        volume_mean = current_data['volume'].iloc[i-19:i+1].mean()
        volume_std = current_data['volume'].iloc[i-19:i+1].std()
        current_volume = current_data['volume'].iloc[i]
        
        if volume_std > 0:
            z_score = (current_volume - volume_mean) / volume_std
            if z_score > 2.5:
                volume_multiplier = 2.5
            elif z_score > 1.5:
                volume_multiplier = 1.8
            elif z_score > 0.8:
                volume_multiplier = 1.3
            else:
                volume_multiplier = 1.0
        else:
            volume_multiplier = 1.0
        
        # 4. Price-Level Proximity Assessment
        high_20 = current_data['high'].iloc[i-19:i+1].max()
        low_20 = current_data['low'].iloc[i-19:i+1].min()
        current_close = current_data['close'].iloc[i]
        
        if high_20 != low_20:
            position = (current_close - low_20) / (high_20 - low_20)
        else:
            position = 0.5
        
        if position > 0.9:
            position_adjustment = 0.7
        elif position >= 0.75:
            position_adjustment = 0.85
        elif position >= 0.25:
            position_adjustment = 1.0
        elif position >= 0.1:
            position_adjustment = 1.15
        else:
            position_adjustment = 1.3
        
        # 5. Final Alpha Construction
        weighted_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        volume_enhanced = weighted_divergence * volume_multiplier
        context_adjusted = volume_enhanced * position_adjustment
        
        alpha.iloc[i] = context_adjusted
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
