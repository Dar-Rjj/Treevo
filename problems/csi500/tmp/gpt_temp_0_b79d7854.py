import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on multi-timeframe momentum-volume divergence
    with volatility regime weighting, volume spike confirmation, and price level context.
    """
    # Calculate returns for volatility assessment
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
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
        vol_5 = returns.iloc[i-4:i+1].std() if i >= 4 else 0
        vol_20 = returns.iloc[i-19:i+1].std()
        
        # Regime-Based Weights
        if vol_5 > vol_20 * 1.1:  # High volatility
            weight_5, weight_20 = 0.2, 0.8
        elif vol_20 * 0.9 <= vol_5 <= vol_20 * 1.1:  # Normal volatility
            weight_5, weight_20 = 0.5, 0.5
        else:  # Low volatility
            weight_5, weight_20 = 0.8, 0.2
        
        # 3. Volume Spike Confirmation
        # Volume Baseline
        volume_ma_20 = current_data['volume'].iloc[i-19:i+1].mean()
        volume_threshold = 2 * volume_ma_20
        
        # Spike Detection
        current_volume = current_data['volume'].iloc[i]
        volume_confirmation = 1.5 if current_volume > volume_threshold else 1.0
        
        # 4. Price Level Context
        # Relative Price Position
        high_20 = current_data['high'].iloc[i-19:i+1].max()
        low_20 = current_data['low'].iloc[i-19:i+1].min()
        current_close = current_data['close'].iloc[i]
        
        if high_20 != low_20:
            price_position = (current_close - low_20) / (high_20 - low_20)
        else:
            price_position = 0.5
        
        # Position Adjustment
        if price_position > 0.8:
            price_multiplier = 0.7
        elif price_position >= 0.3:
            price_multiplier = 1.0
        else:
            price_multiplier = 1.3
        
        # 5. Final Alpha Factor
        # Weighted Divergence
        weighted_5_day = divergence_5 * weight_5
        weighted_20_day = divergence_20 * weight_20
        
        # Combined Divergence
        combined_divergence = weighted_5_day + weighted_20_day
        
        # Volume Adjusted
        volume_adjusted = combined_divergence * volume_confirmation
        
        # Final Factor
        final_factor = volume_adjusted * price_multiplier
        
        alpha_factor.iloc[i] = final_factor
    
    return alpha_factor
