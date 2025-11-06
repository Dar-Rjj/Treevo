import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Acceleration with Volume-Efficiency Adaptive Divergence
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Momentum Acceleration Analysis
        # Short-Term Acceleration (1-3 days)
        short_term_momentum_3 = (current_data['close'].iloc[-1] / current_data['close'].iloc[-4] - 1) * 100
        short_term_momentum_2 = (current_data['close'].iloc[-1] / current_data['close'].iloc[-3] - 1) * 100
        short_term_momentum_1 = (current_data['close'].iloc[-1] / current_data['close'].iloc[-2] - 1) * 100
        
        # Calculate acceleration as change in momentum
        accel_2_1 = short_term_momentum_1 - short_term_momentum_2
        accel_3_2 = short_term_momentum_2 - short_term_momentum_3
        short_term_acceleration = (accel_2_1 + accel_3_2) / 2
        
        # Medium-Term Momentum Context (5-10 days)
        medium_term_momentum_10 = (current_data['close'].iloc[-1] / current_data['close'].iloc[-11] - 1) * 100
        medium_term_momentum_5 = (current_data['close'].iloc[-1] / current_data['close'].iloc[-6] - 1) * 100
        
        # Medium-term acceleration
        medium_term_acceleration = medium_term_momentum_5 - medium_term_momentum_10
        
        # Volume-Efficiency Regime Detection
        recent_volume = current_data['volume'].iloc[-5:]
        recent_range = (current_data['high'].iloc[-5:] - current_data['low'].iloc[-5:]) / current_data['close'].iloc[-5:]
        
        # Volume compression/expansion analysis
        volume_std = recent_volume.std()
        volume_mean = recent_volume.mean()
        volume_cv = volume_std / volume_mean if volume_mean > 0 else 0
        
        # Volume efficiency (price movement per unit volume)
        recent_price_change = abs(current_data['close'].iloc[-1] / current_data['close'].iloc[-6] - 1)
        recent_volume_sum = recent_volume.sum()
        volume_efficiency = recent_price_change / recent_volume_sum if recent_volume_sum > 0 else 0
        
        # Volatility context
        recent_volatility = recent_range.mean()
        volatility_regime = 'high' if recent_volatility > recent_range.quantile(0.7) else 'low'
        
        # Volume-volatility regime classification
        if volume_cv > recent_volume.pct_change().std() and volatility_regime == 'high':
            volume_efficiency_regime = 'high_efficiency'
        elif volume_cv < recent_volume.pct_change().std() * 0.5 and volatility_regime == 'low':
            volume_efficiency_regime = 'low_efficiency'
        else:
            volume_efficiency_regime = 'normal'
        
        # Adaptive Divergence Detection
        current_volume = current_data['volume'].iloc[-1]
        volume_spike_intensity = current_volume / recent_volume.mean() if recent_volume.mean() > 0 else 1
        
        if volume_efficiency_regime == 'high_efficiency':
            # High volume-efficiency regime analysis
            gap_reversal_potential = 0
            if current_data['open'].iloc[-1] > current_data['close'].iloc[-2]:
                # Gap up - check for reversal
                intraday_reversal = (current_data['high'].iloc[-1] - current_data['close'].iloc[-1]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1])
                gap_reversal_potential = intraday_reversal if not np.isnan(intraday_reversal) else 0
            elif current_data['open'].iloc[-1] < current_data['close'].iloc[-2]:
                # Gap down - check for reversal
                intraday_reversal = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1])
                gap_reversal_potential = intraday_reversal if not np.isnan(intraday_reversal) else 0
            
            explosive_move_probability = volume_spike_intensity * (1 - gap_reversal_potential)
            divergence_strength = explosive_move_probability
            
        else:  # Low efficiency or normal regime
            # Volume accumulation patterns
            volume_accumulation = 0
            if len(current_data) >= 15:
                recent_15_volume = current_data['volume'].iloc[-15:]
                volume_trend = np.polyfit(range(15), recent_15_volume, 1)[0]
                volume_accumulation = volume_trend / recent_15_volume.mean() if recent_15_volume.mean() > 0 else 0
            
            # Price compression breakout potential
            recent_ranges = (current_data['high'].iloc[-10:] - current_data['low'].iloc[-10:]) / current_data['close'].iloc[-10:]
            range_compression = recent_ranges.std() / recent_ranges.mean() if recent_ranges.mean() > 0 else 0
            
            accumulation_strength = volume_accumulation * (1 - range_compression)
            divergence_strength = accumulation_strength
        
        # Integrated Signal Generation
        # Multi-Timeframe Acceleration Quality
        acceleration_persistence = 0
        if len(current_data) >= 8:
            # Check if acceleration is consistent over recent periods
            recent_accels = []
            for j in range(2, 6):
                if i - j >= 0:
                    mom_j = (current_data['close'].iloc[-j] / current_data['close'].iloc[-j-3] - 1) * 100
                    mom_j_1 = (current_data['close'].iloc[-j] / current_data['close'].iloc[-j-2] - 1) * 100
                    recent_accels.append(mom_j_1 - mom_j)
            
            if len(recent_accels) > 1:
                acceleration_persistence = np.mean([1 if acc * short_term_acceleration > 0 else -1 for acc in recent_accels])
        
        # Medium-term context alignment
        context_alignment = 1 if short_term_acceleration * medium_term_acceleration > 0 else -1
        
        # Acceleration magnitude scoring
        acceleration_magnitude = abs(short_term_acceleration) + abs(medium_term_acceleration) * 0.5
        
        # Volume-Efficiency Adaptive Weighting
        if volume_efficiency_regime == 'high_efficiency':
            regime_weight = 1.5
        elif volume_efficiency_regime == 'low_efficiency':
            regime_weight = 0.8
        else:
            regime_weight = 1.0
        
        # Volume-volatility confirmation quality
        volatility_confirmation = 1 if (short_term_acceleration > 0 and recent_volatility > recent_range.median()) or \
                                     (short_term_acceleration < 0 and recent_volatility < recent_range.median()) else 0.5
        
        # Final factor combination
        acceleration_quality = (acceleration_persistence + context_alignment) * acceleration_magnitude
        final_factor = acceleration_quality * divergence_strength * regime_weight * volatility_confirmation
        
        result.iloc[i] = final_factor
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
