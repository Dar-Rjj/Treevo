import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe momentum-volume divergence with adaptive weighting
    and volume/intraday quality enhancements.
    """
    # Calculate returns for volatility
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Multi-Timeframe Momentum-Volume Divergence
        # 5-Day calculations
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
        
        # 20-Day calculations
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        divergence_20 = price_momentum_20 - volume_momentum_20
        
        # 2. Volatility Regime Weighting
        # Volatility calculations
        if i >= 5:
            vol_5 = returns.iloc[i-4:i+1].std()
        else:
            vol_5 = returns.iloc[:i+1].std()
        
        vol_20 = returns.iloc[i-19:i+1].std()
        
        # Adaptive weight assignment
        if vol_20 == 0:
            weight_5, weight_20 = 0.4, 0.6
        elif vol_5 > 1.5 * vol_20:  # High volatility regime
            weight_5, weight_20 = 0.8, 0.2
        elif vol_5 < 0.67 * vol_20:  # Low volatility regime
            weight_5, weight_20 = 0.2, 0.8
        else:  # Normal volatility regime
            weight_5, weight_20 = 0.4, 0.6
        
        # 3. Statistical Volume Spike Detection
        volume_window = current_data['volume'].iloc[i-19:i+1]
        volume_median = volume_window.median()
        volume_mad = (volume_window - volume_median).abs().median()
        
        current_volume = current_data['volume'].iloc[i]
        
        if volume_mad == 0:
            volume_multiplier = 1.0
        elif current_volume > volume_median + 3 * volume_mad:
            volume_multiplier = 2.0
        elif current_volume > volume_median + 2 * volume_mad:
            volume_multiplier = 1.5
        else:
            volume_multiplier = 1.0
        
        # 4. Intraday Price Action Quality
        high = current_data['high'].iloc[i]
        low = current_data['low'].iloc[i]
        close = current_data['close'].iloc[i]
        
        if high != low:
            range_efficiency = (close - low) / (high - low)
            volatility_adjusted_range = (high - low) / close
        else:
            range_efficiency = 0.5
            volatility_adjusted_range = 0
        
        # Intraday quality score
        if range_efficiency > 0.7 and volatility_adjusted_range > 0.015:
            intraday_score = 1.4
        elif range_efficiency > 0.6 and volatility_adjusted_range > 0.01:
            intraday_score = 1.2
        elif range_efficiency < 0.4:
            intraday_score = 0.8
        else:
            intraday_score = 1.0
        
        # 5. Robust Alpha Construction
        blended_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        volume_enhanced = blended_divergence * volume_multiplier
        final_alpha = volume_enhanced * intraday_score
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
