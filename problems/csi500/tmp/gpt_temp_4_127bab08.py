import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Regime-Adaptive Multi-Timeframe Momentum Consistency factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Required lookback period
    lookback = 20
    
    for i in range(lookback, len(data)):
        current_data = data.iloc[:i+1]
        
        # Raw Momentum Calculation
        price_moments = []
        volume_moments = []
        
        # Price momentum across timeframes
        for period in [2, 5, 10, 20]:
            if i >= period:
                price_mom = (current_data['close'].iloc[i] / current_data['close'].iloc[i-period]) - 1
            else:
                price_mom = 0
            price_moments.append(price_mom)
        
        # Volume momentum across timeframes
        for period in [2, 5, 10, 20]:
            if i >= period:
                volume_mom = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-period]) - 1
            else:
                volume_mom = 0
            volume_moments.append(volume_mom)
        
        # Regime Detection
        # Market Participation Regime
        if i >= 20:
            amount_intensity = current_data['amount'].iloc[i] / (current_data['amount'].iloc[i-20:i].mean())
            amount_momentum = (current_data['amount'].iloc[i] / current_data['amount'].iloc[i-5]) - 1 if i >= 5 else 0
            
            if amount_intensity > 1.1 and amount_momentum > 0.05:
                participation_regime = 'high'
            elif 0.9 <= amount_intensity <= 1.1:
                participation_regime = 'normal'
            elif amount_intensity < 0.9 and amount_momentum < -0.05:
                participation_regime = 'low'
            else:
                participation_regime = 'normal'
        else:
            participation_regime = 'normal'
            amount_intensity = 1.0
            amount_momentum = 0.0
        
        # Volatility Regime
        if i >= 20:
            daily_range_ratio = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i]
            range_intensity = daily_range_ratio / (current_data['high'].iloc[i-20:i].subtract(current_data['low'].iloc[i-20:i]).divide(current_data['close'].iloc[i-20:i]).mean())
            
            if range_intensity > 1.15:
                volatility_regime = 'high'
            elif 0.85 <= range_intensity <= 1.15:
                volatility_regime = 'normal'
            else:
                volatility_regime = 'low'
        else:
            volatility_regime = 'normal'
            range_intensity = 1.0
        
        # Momentum Consistency Assessment
        # Directional Consistency
        price_direction_agreement = sum(1 for mom in price_moments if mom > 0)
        volume_direction_agreement = sum(1 for mom in volume_moments if mom > 0)
        cross_asset_agreement = price_direction_agreement * volume_direction_agreement
        
        # Magnitude Consistency
        price_moments_abs = [abs(mom) for mom in price_moments if mom != 0]
        volume_moments_abs = [abs(mom) for mom in volume_moments if mom != 0]
        
        if len(price_moments_abs) > 1 and np.mean(price_moments_abs) > 0:
            price_magnitude_stability = 1 - (np.std(price_moments) / np.mean(price_moments_abs))
        else:
            price_magnitude_stability = 0.5
            
        if len(volume_moments_abs) > 1 and np.mean(volume_moments_abs) > 0:
            volume_magnitude_stability = 1 - (np.std(volume_moments) / np.mean(volume_moments_abs))
        else:
            volume_magnitude_stability = 0.5
            
        combined_stability = (price_magnitude_stability + volume_magnitude_stability) / 2
        
        # Dynamic Regime-Adaptive Weighting
        # Timeframe weights by regime
        participation_weights = {
            'high': [0.4, 0.3, 0.2, 0.1],
            'normal': [0.25, 0.3, 0.25, 0.2],
            'low': [0.1, 0.2, 0.3, 0.4]
        }
        
        volatility_weights = {
            'high': [0.35, 0.3, 0.2, 0.15],
            'normal': [0.25, 0.3, 0.25, 0.2],
            'low': [0.15, 0.25, 0.3, 0.3]
        }
        
        part_weight = np.array(participation_weights[participation_regime])
        vol_weight = np.array(volatility_weights[volatility_regime])
        final_weights = (part_weight + vol_weight) / 2
        
        # Alpha Signal Generation
        # Weighted Momentum Score
        price_component = np.sum(np.array(final_weights) * np.array(price_moments))
        volume_component = np.sum(np.array(final_weights) * np.array(volume_moments))
        base_alpha = price_component * volume_component
        
        # Consistency Enhancement
        directional_multiplier = cross_asset_agreement / 16  # Max 16 (4×4)
        stability_multiplier = combined_stability
        
        enhanced_alpha = base_alpha * directional_multiplier * stability_multiplier
        
        # Store result
        result.iloc[i] = enhanced_alpha
    
    # Fill early values with 0
    result = result.fillna(0)
    
    return result
