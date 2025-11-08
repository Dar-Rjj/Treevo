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
        
        # 1. Momentum-Volume Divergence Analysis
        # 5-Day Components
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
        
        # 20-Day Components
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        divergence_20 = price_momentum_20 - volume_momentum_20
        
        # 2. Dynamic Volatility Weighting
        # Volatility Assessment
        vol_5 = returns.iloc[i-4:i+1].std() if i >= 4 else 0
        vol_20 = returns.iloc[i-19:i+1].std()
        vol_ratio = vol_5 / vol_20 if vol_20 != 0 else 1
        
        # Adaptive Timeframe Weights
        if vol_ratio > 1.5:
            weight_5, weight_20 = 0.8, 0.2
        elif vol_ratio >= 0.67:
            weight_5, weight_20 = 0.5, 0.5
        else:
            weight_5, weight_20 = 0.2, 0.8
        
        # 3. Statistical Volume Spike Detection
        # Volume Distribution Analysis
        volume_window = current_data['volume'].iloc[i-19:i+1]
        vol_mean = volume_window.mean()
        vol_std = volume_window.std()
        current_volume = current_data['volume'].iloc[i]
        
        # Adaptive Spike Thresholds
        if vol_std > 0:
            if current_volume > vol_mean + 3 * vol_std:
                volume_multiplier = 2.5
            elif current_volume > vol_mean + 2 * vol_std:
                volume_multiplier = 1.8
            elif current_volume > vol_mean + vol_std:
                volume_multiplier = 1.3
            else:
                volume_multiplier = 1.0
        else:
            volume_multiplier = 1.0
        
        # 4. Price-Level Proximity Assessment
        # Key Price Levels
        high_20 = current_data['high'].iloc[i-19:i+1].max()
        low_20 = current_data['low'].iloc[i-19:i+1].min()
        current_close = current_data['close'].iloc[i]
        
        # Position-Based Adjustment
        if current_close > 0.9 * high_20:
            position_multiplier = 0.7
        elif current_close < 1.1 * low_20:
            position_multiplier = 1.3
        else:
            position_multiplier = 1.0
        
        # 5. Final Alpha Construction
        # Weighted Divergence Blend
        weighted_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        
        # Volume Enhanced Factor
        volume_enhanced_factor = weighted_divergence * volume_multiplier
        
        # Context Finalized Alpha
        final_alpha = volume_enhanced_factor * position_multiplier
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
