import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on multi-timeframe price-volume divergence with dynamic weighting
    and various market condition adjustments.
    """
    # Calculate returns for volatility assessment
    returns = df['close'].pct_change()
    
    # Initialize result series
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(30, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Price-Volume Divergence
        # Short-Term (3-Day) Analysis
        if i >= 3:
            price_momentum_3d = current_data['close'].iloc[i] / current_data['close'].iloc[i-3]
            volume_momentum_3d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3]
            divergence_3d = price_momentum_3d - volume_momentum_3d
        else:
            divergence_3d = 0
            
        # Medium-Term (10-Day) Analysis
        if i >= 10:
            price_momentum_10d = current_data['close'].iloc[i] / current_data['close'].iloc[i-10]
            volume_momentum_10d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-10]
            divergence_10d = price_momentum_10d - volume_momentum_10d
        else:
            divergence_10d = 0
            
        # Long-Term (30-Day) Analysis
        if i >= 30:
            price_momentum_30d = current_data['close'].iloc[i] / current_data['close'].iloc[i-30]
            volume_momentum_30d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-30]
            divergence_30d = price_momentum_30d - volume_momentum_30d
        else:
            divergence_30d = 0
        
        # Volatility-Regime Dynamic Weighting
        # Volatility Assessment
        if i >= 2:
            short_term_vol = returns.iloc[i-2:i+1].std() if len(returns.iloc[i-2:i+1]) > 1 else 0
        else:
            short_term_vol = 0
            
        if i >= 9:
            medium_term_vol = returns.iloc[i-9:i+1].std() if len(returns.iloc[i-9:i+1]) > 1 else 0
        else:
            medium_term_vol = 0
        
        # Volatility Regime Classification
        avg_vol = (short_term_vol + medium_term_vol) / 2 if (short_term_vol > 0 and medium_term_vol > 0) else 0
        
        if avg_vol > 0.03:  # High Volatility Regime
            short_weight, medium_weight, long_weight = 0.6, 0.3, 0.1
        elif avg_vol > 0.015:  # Normal Volatility Regime
            short_weight, medium_weight, long_weight = 0.3, 0.4, 0.3
        else:  # Low Volatility Regime
            short_weight, medium_weight, long_weight = 0.1, 0.3, 0.6
        
        # Weighted Divergence Score
        weighted_divergence = (divergence_3d * short_weight + 
                              divergence_10d * medium_weight + 
                              divergence_30d * long_weight)
        
        # Volume Spike Multiplier System
        if i >= 20:
            volume_ma = current_data['volume'].iloc[i-19:i+1].mean()
            volume_std = current_data['volume'].iloc[i-19:i+1].std()
            current_volume = current_data['volume'].iloc[i]
            
            if current_volume > (volume_ma + 2.5 * volume_std):
                volume_multiplier = 2.0
            elif current_volume > (volume_ma + 1.5 * volume_std):
                volume_multiplier = 1.5
            elif current_volume > (volume_ma + 0.5 * volume_std):
                volume_multiplier = 1.2
            else:
                volume_multiplier = 1.0
        else:
            volume_multiplier = 1.0
        
        # Volume Enhanced Score
        volume_enhanced_score = weighted_divergence * volume_multiplier
        
        # Price Range Position Factor
        if i >= 20:
            high_20d = current_data['high'].iloc[i-19:i+1].max()
            low_20d = current_data['low'].iloc[i-19:i+1].min()
            current_close = current_data['close'].iloc[i]
            
            if high_20d != low_20d:
                position_ratio = (current_close - low_20d) / (high_20d - low_20d)
            else:
                position_ratio = 0.5
                
            if position_ratio > 0.9:
                position_multiplier = 0.5
            elif position_ratio > 0.8:
                position_multiplier = 0.7
            elif position_ratio < 0.1:
                position_multiplier = 1.5
            elif position_ratio < 0.2:
                position_multiplier = 1.3
            else:
                position_multiplier = 1.0
        else:
            position_multiplier = 1.0
        
        # Range Adjusted Score
        range_adjusted_score = volume_enhanced_score * position_multiplier
        
        # Intraday Strength Confirmation
        current_open = current_data['open'].iloc[i]
        current_high = current_data['high'].iloc[i]
        current_low = current_data['low'].iloc[i]
        current_close = current_data['close'].iloc[i]
        
        if current_high != current_low:
            range_efficiency = (current_close - current_open) / (current_high - current_low)
            close_strength = (current_close - current_low) / (current_high - current_low)
        else:
            range_efficiency = 0
            close_strength = 0.5
        
        if range_efficiency > 0.6 and close_strength > 0.7:
            intraday_multiplier = 1.2
        elif range_efficiency < -0.6 and close_strength < 0.3:
            intraday_multiplier = 0.8
        else:
            intraday_multiplier = 1.0
        
        # Intraday Confirmed Score
        intraday_confirmed_score = range_adjusted_score * intraday_multiplier
        
        # Volatility-Adjusted Sensitivity
        if i >= 10:
            recent_vol = returns.iloc[i-9:i+1].std() if len(returns.iloc[i-9:i+1]) > 1 else 0
        else:
            recent_vol = 0
            
        if recent_vol > 0.025:
            volatility_sensitivity = 1.4
        elif recent_vol > 0.015:
            volatility_sensitivity = 1.2
        elif recent_vol >= 0.008:
            volatility_sensitivity = 1.0
        else:
            volatility_sensitivity = 0.7
        
        # Final Alpha Factor
        final_alpha = intraday_confirmed_score * volatility_sensitivity
        alpha_factor.iloc[i] = final_alpha
    
    # Fill NaN values with 0
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
