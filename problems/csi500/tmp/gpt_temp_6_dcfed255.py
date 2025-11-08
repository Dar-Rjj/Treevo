import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe price-volume divergence with dynamic weighting
    and contextual adjustments for volume spikes, price levels, and intraday strength.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns for volatility assessment
    returns = df['close'].pct_change()
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Price-Volume Divergence
        # Short-Term (5-Day)
        if i >= 5:
            price_momentum_5d = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            volume_momentum_5d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]
            divergence_5d = price_momentum_5d - volume_momentum_5d
        else:
            divergence_5d = 0
        
        # Medium-Term (20-Day)
        price_momentum_20d = current_data['close'].iloc[i] / current_data['close'].iloc[i-20]
        volume_momentum_20d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20]
        divergence_20d = price_momentum_20d - volume_momentum_20d
        
        # Volatility-Regime Weighting
        # Volatility Assessment
        short_term_vol = returns.iloc[max(0, i-4):i+1].std() if i >= 4 else 0.01
        medium_term_vol = returns.iloc[max(0, i-19):i+1].std()
        
        if medium_term_vol == 0:
            volatility_ratio = 1.0
        else:
            volatility_ratio = short_term_vol / medium_term_vol
        
        # Dynamic Weight Assignment
        if volatility_ratio > 1.5:
            short_weight, medium_weight = 0.7, 0.3
        elif volatility_ratio >= 0.7:
            short_weight, medium_weight = 0.5, 0.5
        else:
            short_weight, medium_weight = 0.3, 0.7
        
        # Volume Spike Detection
        volume_window = current_data['volume'].iloc[max(0, i-19):i+1]
        volume_ma = volume_window.mean()
        volume_std = volume_window.std()
        current_volume = current_data['volume'].iloc[i]
        
        if volume_std == 0:
            volume_multiplier = 1.0
        else:
            z_score = (current_volume - volume_ma) / volume_std
            if z_score > 2:
                volume_multiplier = 2.0
            elif z_score > 1:
                volume_multiplier = 1.5
            else:
                volume_multiplier = 1.0
        
        # Price-Level Context
        high_20d = current_data['high'].iloc[max(0, i-19):i+1].max()
        low_20d = current_data['low'].iloc[max(0, i-19):i+1].min()
        current_close = current_data['close'].iloc[i]
        
        if current_close > 0.98 * high_20d:
            position_multiplier = 0.7
        elif current_close < 1.02 * low_20d:
            position_multiplier = 1.3
        else:
            position_multiplier = 1.0
        
        # Intraday Strength
        current_high = current_data['high'].iloc[i]
        current_low = current_data['low'].iloc[i]
        
        if current_high - current_low == 0:
            close_position = 0.5
        else:
            close_position = (current_close - current_low) / (current_high - current_low)
        
        if close_position > 0.8:
            intraday_multiplier = 1.2
        elif close_position < 0.2:
            intraday_multiplier = 0.8
        else:
            intraday_multiplier = 1.0
        
        # Final Alpha Construction
        blended_divergence = (divergence_5d * short_weight) + (divergence_20d * medium_weight)
        volume_enhanced = blended_divergence * volume_multiplier
        level_adjusted = volume_enhanced * position_multiplier
        intraday_confirmed = level_adjusted * intraday_multiplier
        
        alpha.iloc[i] = intraday_confirmed
    
    # Fill NaN values with 0 for the first 20 days
    alpha = alpha.fillna(0)
    
    return alpha
