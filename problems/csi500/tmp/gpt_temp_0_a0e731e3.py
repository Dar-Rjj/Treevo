import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on momentum-volume divergence with adaptive weighting,
    volume spike detection, and intraday price efficiency.
    """
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Momentum-Volume Divergence Components
        # 5-Day Component
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
        
        # 20-Day Component
        price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20] - 1
        volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20] - 1
        divergence_20 = price_momentum_20 - volume_momentum_20
        
        # 2. Volatility Regime Detection
        if i >= 20:
            # 5-Day Volatility (last 5 days including current)
            vol_5 = returns.iloc[i-4:i+1].std() if i >= 4 else returns.iloc[:i+1].std()
            # 20-Day Volatility (last 20 days including current)
            vol_20 = returns.iloc[i-19:i+1].std()
            
            if vol_5 > 1.5 * vol_20:
                # High Volatility Regime
                weight_5 = 0.7
                weight_20 = 0.3
            elif vol_5 >= 0.67 * vol_20:
                # Normal Volatility Regime
                weight_5 = 0.5
                weight_20 = 0.5
            else:
                # Low Volatility Regime
                weight_5 = 0.3
                weight_20 = 0.7
        else:
            weight_5 = 0.5
            weight_20 = 0.5
        
        # 3. Volume Spike Detection
        volume_window = current_data['volume'].iloc[i-19:i+1]
        volume_median = volume_window.median()
        volume_mad = (volume_window - volume_median).abs().median()
        
        current_volume = current_data['volume'].iloc[i]
        if current_volume > volume_median + 3 * volume_mad:
            volume_multiplier = 2.0
        elif current_volume > volume_median + 2 * volume_mad:
            volume_multiplier = 1.5
        elif current_volume > volume_median + volume_mad:
            volume_multiplier = 1.2
        else:
            volume_multiplier = 1.0
        
        # 4. Intraday Price Efficiency
        high = current_data['high'].iloc[i]
        low = current_data['low'].iloc[i]
        close = current_data['close'].iloc[i]
        
        daily_range = (high - low) / close
        close_position = (close - low) / (high - low) if high != low else 0.5
        
        if close_position > 0.7 and daily_range > 0.02:
            efficiency_multiplier = 1.3
        elif close_position > 0.5 and daily_range > 0.01:
            efficiency_multiplier = 1.1
        else:
            efficiency_multiplier = 0.9
        
        # 5. Composite Alpha Factor
        weighted_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        volume_enhanced = weighted_divergence * volume_multiplier
        final_alpha = volume_enhanced * efficiency_multiplier
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
