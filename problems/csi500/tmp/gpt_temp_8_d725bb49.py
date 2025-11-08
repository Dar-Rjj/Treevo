import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe momentum-volume divergence with dynamic weighting
    and price level adjustments.
    """
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
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
        
        # Volatility Assessment
        if i >= 5:
            vol_5 = returns.iloc[i-4:i+1].std()
        else:
            vol_5 = returns.iloc[:i+1].std()
        
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
        volume_20d_avg = current_data['volume'].iloc[i-19:i+1].mean()
        volume_20d_std = current_data['volume'].iloc[i-19:i+1].std()
        volume_zscore = (current_data['volume'].iloc[i] - volume_20d_avg) / volume_20d_std if volume_20d_std != 0 else 0
        spike_multiplier = 1 + (0.3 * max(0, volume_zscore - 1))
        
        # Key Price Level Proximity
        high_20d = current_data['high'].iloc[i-19:i+1].max()
        low_20d = current_data['low'].iloc[i-19:i+1].min()
        price_position = (current_data['close'].iloc[i] - low_20d) / (high_20d - low_20d) if (high_20d - low_20d) != 0 else 0.5
        
        if price_position > 0.85:
            adjustment_factor = 0.7
        elif price_position < 0.15:
            adjustment_factor = 1.3
        else:
            adjustment_factor = 1.0
        
        # Dynamic Sensitivity Adaptation
        recent_trend = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] if i >= 5 else 1.0
        volatility_component = 1 / (1 + vol_5) if vol_5 != 0 else 1.0
        trend_component = 1 + 0.2 * np.sign(recent_trend - 1)
        combined_sensitivity = volatility_component * trend_component
        
        # Final Alpha Construction
        blended_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        volume_enhanced = blended_divergence * spike_multiplier
        price_level_adjusted = volume_enhanced * adjustment_factor
        dynamic_final_alpha = price_level_adjusted * combined_sensitivity
        
        alpha.iloc[i] = dynamic_final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
