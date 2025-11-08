import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe momentum-volume divergence with dynamic regime weighting
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required lookback periods
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Momentum-Volume Divergence
        # 5-Day Framework
        if i >= 5:
            price_momentum_5 = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
            volume_momentum_5 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1
            divergence_5 = price_momentum_5 - volume_momentum_5
        else:
            divergence_5 = 0
        
        # 20-Day Framework
        if i >= 20:
            price_momentum_20 = current_data['close'].iloc[i] / current_data['close'].iloc[i-20] - 1
            volume_momentum_20 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20] - 1
            divergence_20 = price_momentum_20 - volume_momentum_20
        else:
            divergence_20 = 0
        
        # Dynamic Volatility Regime Weighting
        # Volatility Calculation
        if i >= 5:
            returns_5 = np.log(current_data['close'].iloc[i-4:i+1] / current_data['close'].iloc[i-5:i].values)
            volatility_5 = returns_5.std()
        else:
            volatility_5 = 0
            
        if i >= 20:
            returns_20 = np.log(current_data['close'].iloc[i-19:i+1] / current_data['close'].iloc[i-20:i].values)
            volatility_20 = returns_20.std()
        else:
            volatility_20 = volatility_5 if volatility_5 > 0 else 0.01
        
        volatility_ratio = volatility_5 / volatility_20 if volatility_20 > 0 else 1.0
        
        # Adaptive Weight Assignment
        if volatility_ratio > 2.0:
            weight_5, weight_20 = 0.9, 0.1
        elif volatility_ratio >= 1.2:
            weight_5, weight_20 = 0.7, 0.3
        elif volatility_ratio >= 0.8:
            weight_5, weight_20 = 0.5, 0.5
        elif volatility_ratio >= 0.5:
            weight_5, weight_20 = 0.3, 0.7
        else:
            weight_5, weight_20 = 0.1, 0.9
        
        # Advanced Volume Regime Analysis
        if i >= 20:
            volume_window = current_data['volume'].iloc[i-19:i+1]
            volume_mean = volume_window.mean()
            volume_std = volume_window.std()
            volume_zscore = (current_data['volume'].iloc[i] - volume_mean) / volume_std if volume_std > 0 else 0
        else:
            volume_zscore = 0
        
        # Volume Regime Classification
        if volume_zscore > 3.0:
            volume_multiplier = 3.0
        elif volume_zscore > 2.0:
            volume_multiplier = 2.0
        elif volume_zscore > 1.5:
            volume_multiplier = 1.5
        elif volume_zscore < -1.5:
            volume_multiplier = 0.7
        else:
            volume_multiplier = 1.0
        
        # Intraday Price Action Dynamics
        # Range Efficiency Analysis
        daily_range = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i-1] if i > 0 else 0
        close_position = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) > 0 else 0.5
        range_efficiency = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) > 0 else 0
        
        # Gap Analysis
        opening_gap = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1] if i > 0 else 0
        gap_filled = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) if i > 0 and abs(current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) > 0 else 0
        gap_strength = np.sign(opening_gap) * (1 - abs(gap_filled)) if i > 0 else 0
        
        # Price-Level Context Analysis
        if i >= 10:
            recent_high = current_data['high'].iloc[i-9:i+1].max()
            recent_low = current_data['low'].iloc[i-9:i+1].min()
            near_term_position = (current_data['close'].iloc[i] - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5
            
            resistance_distance = (recent_high - current_data['close'].iloc[i]) / current_data['close'].iloc[i]
            support_distance = (current_data['close'].iloc[i] - recent_low) / current_data['close'].iloc[i]
            position_strength = support_distance / (support_distance + resistance_distance) if (support_distance + resistance_distance) > 0 else 0.5
        else:
            near_term_position = 0.5
            position_strength = 0.5
        
        # Regime Shift Detection System
        volatility_breakout = volatility_ratio > 2.5
        volatility_collapse = volatility_ratio < 0.4
        
        volume_acceleration = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] if i >= 5 else 1.0
        volume_regime_break = volume_acceleration > 2.5 and volume_zscore > 2.0
        volume_drying = volume_acceleration < 0.4 and volume_zscore < -1.0
        
        # Final Alpha Construction
        base_divergence_blend = (divergence_5 * weight_5) + (divergence_20 * weight_20)
        volume_enhanced = base_divergence_blend * volume_multiplier
        intraday_adjusted = volume_enhanced * (1 + range_efficiency * close_position)
        gap_momentum_adjusted = intraday_adjusted * (1 + gap_strength * abs(opening_gap))
        position_context_adjusted = gap_momentum_adjusted * (1 + (position_strength - 0.5) * 2)
        
        # Regime Finalized
        if volatility_breakout or volume_regime_break:
            final_alpha = position_context_adjusted * 2.0
        elif volatility_collapse or volume_drying:
            final_alpha = position_context_adjusted * 0.5
        else:
            final_alpha = position_context_adjusted
        
        result.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
