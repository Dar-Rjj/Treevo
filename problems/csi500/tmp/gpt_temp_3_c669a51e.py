import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe momentum-volume divergence with volatility regime weighting,
    volume spike confirmation, and key price level proximity adjustments.
    """
    # Calculate returns for volatility assessment
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Multi-Timeframe Momentum-Volume Divergence
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
        
        # 2. Volatility Regime Weighting
        # Volatility Assessment
        if i >= 30:
            vol_10 = returns.iloc[i-9:i+1].std()
            vol_30 = returns.iloc[i-29:i+1].std()
            
            # Dynamic Weight Assignment
            if vol_10 > 1.5 * vol_30:  # High Volatility
                weight_5, weight_20 = 0.8, 0.2
            elif vol_10 >= vol_30 and vol_10 <= 1.5 * vol_30:  # Normal Volatility
                weight_5, weight_20 = 0.5, 0.5
            else:  # Low Volatility
                weight_5, weight_20 = 0.2, 0.8
        else:
            weight_5, weight_20 = 0.5, 0.5
        
        # 3. Volume Spike Confirmation
        # Volume Baseline
        volume_window = current_data['volume'].iloc[i-19:i+1]
        volume_avg = volume_window.mean()
        volume_std = volume_window.std()
        
        # Spike Detection
        current_volume = current_data['volume'].iloc[i]
        if current_volume > (volume_avg + 2 * volume_std):
            volume_multiplier = 2.0
        else:
            volume_multiplier = 1.0
        
        # 4. Key Price Level Proximity
        # Level Identification
        price_window = current_data['close'].iloc[i-19:i+1]
        high_20 = price_window.max()
        low_20 = price_window.min()
        current_close = current_data['close'].iloc[i]
        
        # Proximity Adjustment
        if current_close > 0.95 * high_20:  # Near Resistance
            proximity_multiplier = 0.7
        elif current_close < 1.05 * low_20:  # Near Support
            proximity_multiplier = 1.3
        else:  # Middle Range
            proximity_multiplier = 1.0
        
        # 5. Adaptive Alpha Construction
        blended_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        volume_enhanced = blended_divergence * volume_multiplier
        context_adjusted = volume_enhanced * proximity_multiplier
        
        alpha.iloc[i] = context_adjusted
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
