import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on multi-timeframe momentum-volume divergence
    with volatility regime weighting, volume spike confirmation, and key price level proximity.
    """
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
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
        if i >= 5:
            vol_5 = returns.iloc[i-4:i+1].std()
        else:
            vol_5 = returns.iloc[:i+1].std()
        
        vol_20 = returns.iloc[i-19:i+1].std()
        
        if vol_20 > 0:
            vol_ratio = vol_5 / vol_20
        else:
            vol_ratio = 1.0
        
        # Dynamic Weight Assignment
        if vol_ratio > 1.5:  # High Volatility
            weight_5 = 0.8
            weight_20 = 0.2
        elif vol_ratio >= 0.67:  # Normal Volatility
            weight_5 = 0.5
            weight_20 = 0.5
        else:  # Low Volatility
            weight_5 = 0.2
            weight_20 = 0.8
        
        # Blended Divergence Score
        blended_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        
        # 3. Volume Spike Confirmation
        # Volume Baseline
        volume_ma_20 = current_data['volume'].iloc[i-19:i+1].mean()
        volume_std_20 = current_data['volume'].iloc[i-19:i+1].std()
        
        current_volume = current_data['volume'].iloc[i]
        
        # Spike Detection
        if current_volume > (volume_ma_20 + 2 * volume_std_20):
            volume_multiplier = 2.0
        elif current_volume > (volume_ma_20 + volume_std_20):
            volume_multiplier = 1.5
        else:
            volume_multiplier = 1.0
        
        # Volume Confirmed Score
        volume_confirmed_score = blended_divergence * volume_multiplier
        
        # 4. Key Price Level Proximity
        # Critical Levels Identification
        high_20 = current_data['high'].iloc[i-19:i+1].max()
        low_20 = current_data['low'].iloc[i-19:i+1].min()
        current_close = current_data['close'].iloc[i]
        
        if high_20 != low_20:
            current_position = (current_close - low_20) / (high_20 - low_20)
        else:
            current_position = 0.5
        
        # Adaptive Sensitivity Adjustment
        if current_position > 0.85:  # Near Resistance
            position_multiplier = 0.7
        elif current_position < 0.15:  # Near Support
            position_multiplier = 1.3
        else:  # Middle Range
            position_multiplier = 1.0
        
        # Final Alpha Construction
        final_score = volume_confirmed_score * position_multiplier
        result.iloc[i] = final_score
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
