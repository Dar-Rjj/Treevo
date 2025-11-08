import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Adaptive Momentum with Volume-Price Alignment alpha factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required lookback periods
    lookbacks = [3, 5, 10, 20]
    max_lookback = max(lookbacks)
    
    # Skip early periods without sufficient data
    for i in range(max_lookback, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Price Momentum
        price_momentum = {}
        for lb in lookbacks:
            if i >= lb:
                price_momentum[lb] = (current_data['close'].iloc[i] / current_data['close'].iloc[i-lb]) - 1
            else:
                price_momentum[lb] = 0
        
        # Volume Momentum Framework
        volume_momentum = {}
        for lb in lookbacks:
            if i >= lb:
                volume_momentum[lb] = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-lb]) - 1
            else:
                volume_momentum[lb] = 0
        
        # Volatility Regime Classification
        daily_range = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i]
        
        range_5 = 0
        range_10 = 0
        range_15 = 0
        if i >= 5:
            range_5 = (current_data['high'].iloc[i-5] - current_data['low'].iloc[i-5]) / current_data['close'].iloc[i-5]
        if i >= 10:
            range_10 = (current_data['high'].iloc[i-10] - current_data['low'].iloc[i-10]) / current_data['close'].iloc[i-10]
        if i >= 15:
            range_15 = (current_data['high'].iloc[i-15] - current_data['low'].iloc[i-15]) / current_data['close'].iloc[i-15]
        
        avg_prev_range = (range_5 + range_10 + range_15) / 3 if (range_5 + range_10 + range_15) > 0 else daily_range
        range_momentum = (daily_range / range_5) - 1 if range_5 > 0 else 0
        
        # Volatility regime states
        if daily_range > avg_prev_range:
            volatility_regime = 'high'
        elif daily_range < avg_prev_range:
            volatility_regime = 'low'
        else:
            volatility_regime = 'transitioning'
        
        if abs(range_momentum) > 0.2:
            volatility_regime = 'transitioning'
        
        # Multi-Timeframe Alignment Analysis
        # Price momentum alignment
        momentum_values = [price_momentum[3], price_momentum[5], price_momentum[10], price_momentum[20]]
        positive_count = sum(1 for m in momentum_values if m > 0)
        momentum_std = np.std(momentum_values) if len(momentum_values) > 1 else 0
        avg_momentum = np.mean(momentum_values)
        
        # Volume-price alignment
        volume_confirmation_count = 0
        volume_divergence_count = 0
        for lb in lookbacks:
            if price_momentum[lb] * volume_momentum[lb] > 0:
                volume_confirmation_count += 1
            elif price_momentum[lb] * volume_momentum[lb] < 0:
                volume_divergence_count += 1
        
        # Cross-timeframe coherence
        ultra_short_short = np.sign(price_momentum[3]) * np.sign(price_momentum[5]) if i >= 5 else 0
        short_medium = np.sign(price_momentum[5]) * np.sign(price_momentum[10]) if i >= 10 else 0
        medium_long = np.sign(price_momentum[10]) * np.sign(price_momentum[20]) if i >= 20 else 0
        
        # Adaptive Weighting System
        # Volatility-based timeframe emphasis
        if volatility_regime == 'high':
            weights = {3: 0.6, 5: 0.3, 10: 0.1, 20: 0.0}
        elif volatility_regime == 'low':
            weights = {3: 0.0, 5: 0.2, 10: 0.4, 20: 0.4}
        else:  # transitioning
            weights = {3: 0.4, 5: 0.4, 10: 0.2, 20: 0.0}
        
        # Volume confirmation weighting
        total_alignments = volume_confirmation_count + volume_divergence_count
        if total_alignments > 0:
            confirmation_ratio = volume_confirmation_count / total_alignments
            if confirmation_ratio > 0.7:
                volume_weight = 1.5  # Strong confirmation
            elif confirmation_ratio < 0.3:
                volume_weight = 0.5  # Strong divergence
            else:
                volume_weight = 1.0  # Neutral
        else:
            volume_weight = 1.0
        
        # Alignment bonus system
        if positive_count == 4 or positive_count == 0:
            alignment_bonus = 2.0  # Perfect alignment
        elif positive_count == 3 or positive_count == 1:
            alignment_bonus = 1.5  # Strong alignment
        else:
            alignment_bonus = 1.0  # Weak alignment
        
        # Volume-Price Divergence Detection
        ultra_short_vp_ratio = 0
        if abs(price_momentum[3]) > 0.001:
            ultra_short_vp_ratio = volume_momentum[3] / price_momentum[3]
        
        divergence_magnitude = abs(volume_momentum[3] - price_momentum[3])
        
        # Final Alpha Construction
        # Base momentum signal
        weighted_momentum = 0
        total_weight = 0
        for lb, weight in weights.items():
            weighted_momentum += price_momentum[lb] * weight
            total_weight += weight
        
        if total_weight > 0:
            base_signal = weighted_momentum / total_weight
        else:
            base_signal = 0
        
        # Apply alignment bonus
        base_signal *= alignment_bonus
        
        # Volume confirmation adjustment
        adjusted_signal = base_signal * volume_weight
        
        # Multi-timeframe coherence enhancement
        coherence_multiplier = 1.0
        positive_coherence = sum([1 for x in [ultra_short_short, short_medium, medium_long] if x > 0])
        if positive_coherence >= 2:
            coherence_multiplier = 1.2
        elif positive_coherence <= 1:
            coherence_multiplier = 0.8
        
        # Final factor
        final_factor = adjusted_signal * coherence_multiplier
        
        result.iloc[i] = final_factor
    
    # Fill early NaN values with 0
    result = result.fillna(0)
    
    return result
