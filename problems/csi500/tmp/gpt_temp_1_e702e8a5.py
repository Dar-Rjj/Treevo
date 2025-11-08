import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Momentum Persistence alpha factor
    """
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate momentum for different timeframes
    for i in range(15, len(df)):
        current_data = df.iloc[:i+1]
        
        # Short-term (3-day) momentum
        if i >= 3:
            short_price_return = (close.iloc[i] / close.iloc[i-3]) - 1
            short_price_dir = np.sign(short_price_return)
            short_price_strength = abs(short_price_return)
            
            short_volume_change = (volume.iloc[i] / volume.iloc[i-3]) - 1
            short_volume_dir = np.sign(short_volume_change)
            short_volume_intensity = abs(short_volume_change)
            
            # Short-term divergence analysis
            if short_price_dir < 0 and short_volume_dir > 0:
                short_divergence = 1  # Bullish divergence
            elif short_price_dir > 0 and short_volume_dir < 0:
                short_divergence = -1  # Bearish divergence
            else:
                short_divergence = 0  # Convergence or neutral
        else:
            short_divergence = 0
            short_price_strength = 0
            short_volume_intensity = 0
        
        # Medium-term (8-day) momentum
        if i >= 8:
            medium_price_return = (close.iloc[i] / close.iloc[i-8]) - 1
            medium_price_dir = np.sign(medium_price_return)
            medium_price_strength = abs(medium_price_return)
            
            medium_volume_change = (volume.iloc[i] / volume.iloc[i-8]) - 1
            medium_volume_dir = np.sign(medium_volume_change)
            medium_volume_intensity = abs(medium_volume_change)
            
            # Medium-term divergence analysis
            if medium_price_dir < 0 and medium_volume_dir > 0:
                medium_divergence = 1  # Bullish divergence
            elif medium_price_dir > 0 and medium_volume_dir < 0:
                medium_divergence = -1  # Bearish divergence
            else:
                medium_divergence = 0  # Convergence or neutral
        else:
            medium_divergence = 0
            medium_price_strength = 0
            medium_volume_intensity = 0
        
        # Long-term (15-day) momentum
        if i >= 15:
            long_price_return = (close.iloc[i] / close.iloc[i-15]) - 1
            long_price_dir = np.sign(long_price_return)
            long_price_strength = abs(long_price_return)
            
            long_volume_change = (volume.iloc[i] / volume.iloc[i-15]) - 1
            long_volume_dir = np.sign(long_volume_change)
            long_volume_intensity = abs(long_volume_change)
            
            # Long-term divergence analysis
            if long_price_dir < 0 and long_volume_dir > 0:
                long_divergence = 1  # Bullish divergence
            elif long_price_dir > 0 and long_volume_dir < 0:
                long_divergence = -1  # Bearish divergence
            else:
                long_divergence = 0  # Convergence or neutral
        else:
            long_divergence = 0
            long_price_strength = 0
            long_volume_intensity = 0
        
        # Multi-timeframe confirmation logic
        divergences = [short_divergence, medium_divergence, long_divergence]
        non_zero_divergences = [d for d in divergences if d != 0]
        
        if len(non_zero_divergences) == 0:
            # No divergence patterns
            factor_value = 0
        else:
            # Calculate divergence consistency
            if len(set(non_zero_divergences)) == 1:
                # Strong confirmation - all non-zero divergences agree
                divergence_type = non_zero_divergences[0]
                consistency_score = 1.0
            elif len(set(non_zero_divergences)) == 2:
                # Moderate confirmation - mixed signals
                divergence_type = np.mean(non_zero_divergences)
                consistency_score = 0.5
            else:
                # Weak confirmation - contradictory signals
                divergence_type = np.mean(non_zero_divergences)
                consistency_score = 0.2
            
            # Calculate momentum strength components
            price_strengths = [short_price_strength, medium_price_strength, long_price_strength]
            volume_intensities = [short_volume_intensity, medium_volume_intensity, long_volume_intensity]
            
            avg_momentum_strength = np.mean([s for s in price_strengths if s > 0])
            avg_volume_intensity = np.mean([v for v in volume_intensities if v > 0])
            
            # Momentum persistence assessment
            if i >= 8:
                short_medium_persistence = 1 if short_price_dir == medium_price_dir else 0
            else:
                short_medium_persistence = 0
                
            if i >= 15:
                medium_long_persistence = 1 if medium_price_dir == long_price_dir else 0
            else:
                medium_long_persistence = 0
            
            persistence_score = (short_medium_persistence + medium_long_persistence) / 2 if (short_medium_persistence + medium_long_persistence) > 0 else 0.1
            
            # Construct final factor value
            base_signal = divergence_type
            strength_weight = avg_momentum_strength if not np.isnan(avg_momentum_strength) else 0
            volume_weight = avg_volume_intensity if not np.isnan(avg_volume_intensity) else 0
            
            factor_value = (base_signal * consistency_score * 
                          (1 + strength_weight) * 
                          (1 + volume_weight) * 
                          (1 + persistence_score))
        
        factor.iloc[i] = factor_value
    
    # Fill initial NaN values with 0
    factor = factor.fillna(0)
    
    return factor
