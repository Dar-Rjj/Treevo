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
        
        # 5-Day Component
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
        
        # 20-Day Component
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        divergence_20 = price_momentum_20 - volume_momentum_20
        
        # Volatility Assessment
        vol_5 = returns.iloc[i-4:i+1].std() if i >= 4 else returns.iloc[:i+1].std()
        vol_20 = returns.iloc[i-19:i+1].std()
        
        # Regime-Based Blending
        if vol_5 > 1.5 * vol_20:
            weight_5, weight_20 = 0.8, 0.2
        elif vol_5 < 0.67 * vol_20:
            weight_5, weight_20 = 0.2, 0.8
        else:
            weight_5, weight_20 = 0.5, 0.5
        
        # Adaptive Volume Spike Detection
        vol_20_data = current_data['volume'].iloc[i-19:i+1]
        vol_median = vol_20_data.median()
        vol_mad = (vol_20_data - vol_median).abs().median()
        
        current_volume = current_data['volume'].iloc[i]
        if current_volume > vol_median + 3 * vol_mad:
            volume_multiplier = 2.0
        elif current_volume > vol_median + 2 * vol_mad:
            volume_multiplier = 1.5
        else:
            volume_multiplier = 1.0
        
        # Intraday Price Efficiency
        high = current_data['high'].iloc[i]
        low = current_data['low'].iloc[i]
        close = current_data['close'].iloc[i]
        
        daily_range = (high - low) / close
        close_position = (close - low) / (high - low) if high != low else 0.5
        
        if close_position > 0.7 and daily_range > 0.02:
            efficiency_score = 1.2
        elif close_position > 0.5 and daily_range > 0.01:
            efficiency_score = 1.0
        else:
            efficiency_score = 0.8
        
        # Composite Alpha Construction
        weighted_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        volume_enhanced = weighted_divergence * volume_multiplier
        final_alpha = volume_enhanced * efficiency_score
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
