import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using momentum-volume divergence with dynamic volatility weighting,
    volume spike detection, and price-level proximity analysis.
    """
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Momentum-Volume Divergence Framework
        # 5-Day Analysis
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
        
        # 20-Day Analysis
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        divergence_20 = price_momentum_20 - volume_momentum_20
        
        # Blended Divergence
        if i >= 5:
            blended_divergence = (divergence_5 + divergence_20) / 2
        else:
            blended_divergence = divergence_20
        
        # 2. Dynamic Volatility Weighting
        # Volatility Assessment
        if i >= 4:
            vol_5 = returns.iloc[i-4:i+1].std()
        else:
            vol_5 = returns.iloc[:i+1].std()
        
        vol_20 = returns.iloc[i-19:i+1].std()
        
        if vol_20 > 0:
            volatility_ratio = vol_5 / vol_20
        else:
            volatility_ratio = 1.0
        
        # Adaptive Weighting Scheme
        if volatility_ratio > 1.5:
            volatility_weight = 0.3
        elif volatility_ratio >= 0.8:
            volatility_weight = 0.5
        else:
            volatility_weight = 0.7
        
        # 3. Statistical Volume Spike Detection
        volume_window = current_data['volume'].iloc[i-19:i+1]
        volume_mean = volume_window.mean()
        volume_std = volume_window.std()
        
        if volume_std > 0:
            volume_zscore = (current_data['volume'].iloc[i] - volume_mean) / volume_std
        else:
            volume_zscore = 0
        
        # Spike-Based Multipliers
        if volume_zscore > 3:
            volume_multiplier = 2.5
        elif volume_zscore > 2:
            volume_multiplier = 1.8
        elif volume_zscore > 1:
            volume_multiplier = 1.3
        else:
            volume_multiplier = 1.0
        
        # 4. Price-Level Proximity Analysis
        price_window = current_data['close'].iloc[i-9:i+1]
        high_10 = price_window.max()
        low_10 = price_window.min()
        
        if high_10 != low_10:
            current_position = (current_data['close'].iloc[i] - low_10) / (high_10 - low_10)
        else:
            current_position = 0.5
        
        # Adaptive Thresholds
        if current_position > 0.9:
            price_adjustment = 0.6
        elif current_position < 0.1:
            price_adjustment = 1.4
        else:
            price_adjustment = 1.0
        
        # 5. Final Alpha Construction
        base_factor = blended_divergence * volatility_weight
        volume_enhanced = base_factor * volume_multiplier
        final_alpha = volume_enhanced * price_adjustment
        
        alpha.iloc[i] = final_alpha
    
    return alpha
