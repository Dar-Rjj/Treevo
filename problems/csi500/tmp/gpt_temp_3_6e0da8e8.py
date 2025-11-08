import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe price-volume divergence with volatility-regime adaptive weighting,
    volume spike confirmation, and price level context adjustment.
    """
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns for volatility calculation
    returns = df['close'].pct_change()
    
    for i in range(60, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Price-Volume Divergence
        # Short-Term (5-Day)
        if i >= 5:
            price_return_5d = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
            volume_return_5d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1
            divergence_5d = price_return_5d - volume_return_5d
        else:
            divergence_5d = 0
        
        # Medium-Term (20-Day)
        if i >= 20:
            price_return_20d = current_data['close'].iloc[i] / current_data['close'].iloc[i-20] - 1
            volume_return_20d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20] - 1
            divergence_20d = price_return_20d - volume_return_20d
        else:
            divergence_20d = 0
        
        # Long-Term (60-Day)
        if i >= 60:
            price_return_60d = current_data['close'].iloc[i] / current_data['close'].iloc[i-60] - 1
            volume_return_60d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-60] - 1
            divergence_60d = price_return_60d - volume_return_60d
        else:
            divergence_60d = 0
        
        # Volatility-Regime Adaptive Weighting
        if i >= 20:
            # Volatility Calculation
            short_term_vol = returns.iloc[max(0, i-4):i+1].std()
            medium_term_vol = returns.iloc[max(0, i-19):i+1].std()
            
            if medium_term_vol > 0:
                volatility_ratio = short_term_vol / medium_term_vol
                
                # Dynamic Weight Assignment
                if volatility_ratio > 1.5:  # High Volatility Regime
                    short_weight, medium_weight, long_weight = 0.6, 0.3, 0.1
                elif volatility_ratio >= 0.8:  # Normal Volatility Regime
                    short_weight, medium_weight, long_weight = 0.3, 0.4, 0.3
                else:  # Low Volatility Regime
                    short_weight, medium_weight, long_weight = 0.1, 0.3, 0.6
            else:
                short_weight, medium_weight, long_weight = 0.3, 0.4, 0.3
        else:
            short_weight, medium_weight, long_weight = 0.3, 0.4, 0.3
        
        # Weighted Divergence
        weighted_divergence = (divergence_5d * short_weight + 
                             divergence_20d * medium_weight + 
                             divergence_60d * long_weight)
        
        # Volume Spike Confirmation System
        if i >= 20:
            # Volume Baseline
            volume_ma_20d = current_data['volume'].iloc[i-19:i+1].mean()
            volume_std_20d = current_data['volume'].iloc[i-19:i+1].std()
            current_volume = current_data['volume'].iloc[i]
            
            # Spike Classification
            if current_volume > (volume_ma_20d + 2 * volume_std_20d):
                spike_multiplier = 2.0
            elif current_volume > (volume_ma_20d + volume_std_20d):
                spike_multiplier = 1.5
            else:
                spike_multiplier = 1.0
            
            # Conditional Application - only apply when divergence is significant
            if abs(weighted_divergence) > 0.02:  # 2% threshold
                volume_enhanced_signal = weighted_divergence * spike_multiplier
            else:
                volume_enhanced_signal = weighted_divergence
        else:
            volume_enhanced_signal = weighted_divergence
        
        # Price Level Context Adjustment
        if i >= 20:
            # Support/Resistance Bands
            high_20d = current_data['high'].iloc[i-19:i+1].max()
            low_20d = current_data['low'].iloc[i-19:i+1].min()
            current_close = current_data['close'].iloc[i]
            
            if high_20d != low_20d:
                position_ratio = (current_close - low_20d) / (high_20d - low_20d)
                
                # Context Multiplier
                if position_ratio > 0.75:  # Resistance Zone
                    context_multiplier = 0.6
                elif position_ratio < 0.25:  # Support Zone
                    context_multiplier = 1.4
                elif position_ratio > 0.5:  # Upper Middle
                    context_multiplier = 0.8
                elif position_ratio < 0.5:  # Lower Middle
                    context_multiplier = 1.2
                else:  # Neutral Zone
                    context_multiplier = 1.0
            else:
                context_multiplier = 1.0
        else:
            context_multiplier = 1.0
        
        # Final Alpha Construction
        final_alpha = volume_enhanced_signal * context_multiplier
        alpha.iloc[i] = final_alpha
    
    # Fill early values with 0
    alpha = alpha.fillna(0)
    
    return alpha
