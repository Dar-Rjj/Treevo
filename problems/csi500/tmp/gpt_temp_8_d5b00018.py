import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum with Volume-Price Persistence alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate required features for each day
    for i in range(9, len(data)):
        current_date = data.index[i]
        
        # Extract current day data
        open_t = data['open'].iloc[i]
        high_t = data['high'].iloc[i]
        low_t = data['low'].iloc[i]
        close_t = data['close'].iloc[i]
        volume_t = data['volume'].iloc[i]
        
        # Extract historical data (t-1 to t-9)
        close_prices = data['close'].iloc[i-9:i+1].values
        volumes = data['volume'].iloc[i-9:i+1].values
        daily_ranges = data['high'].iloc[i-9:i+1].values - data['low'].iloc[i-9:i+1].values
        
        # MOMENTUM PERSISTENCE ANALYSIS
        # Multi-Timeframe Momentum
        mom_1d = close_t - close_prices[-2]  # t-1
        mom_3d = close_t - close_prices[-4]  # t-3
        mom_10d = close_t - close_prices[0]  # t-9
        
        # Momentum Direction Persistence
        recent_moments = [close_prices[j+1] - close_prices[j] for j in range(len(close_prices)-1)]
        recent_mom_signs = [np.sign(mom) for mom in recent_moments]
        
        # Recent direction streak
        direction_streak = 1
        for j in range(len(recent_mom_signs)-1, 0, -1):
            if recent_mom_signs[j] == recent_mom_signs[j-1] and recent_mom_signs[j] != 0:
                direction_streak += 1
            else:
                break
        
        # Direction consistency (last 5 days)
        last_5_signs = recent_mom_signs[-5:]
        positive_returns = sum(1 for sign in last_5_signs if sign > 0)
        direction_consistency = positive_returns / 5.0
        
        # Momentum acceleration
        mom_acceleration = (close_t - close_prices[-4]) - (close_prices[-4] - close_prices[-7])
        
        # Volatility-Adjusted Momentum
        range_volatility_10d = np.sum(daily_ranges)
        range_volatility_3d = np.sum(daily_ranges[-3:])
        
        vam_3d = mom_3d / range_volatility_3d if range_volatility_3d != 0 else 0
        vam_10d = mom_10d / range_volatility_10d if range_volatility_10d != 0 else 0
        
        # VOLUME-PRICE PERSISTENCE
        # Volume Direction Analysis
        volume_changes = [volumes[j+1] - volumes[j] for j in range(len(volumes)-1)]
        volume_signs = [np.sign(change) for change in volume_changes]
        
        # Volume direction streak
        volume_streak = 1
        for j in range(len(volume_signs)-1, 0, -1):
            if volume_signs[j] == volume_signs[j-1] and volume_signs[j] != 0:
                volume_streak += 1
            else:
                break
        
        # Volume persistence strength
        volume_persistence_strength = volume_streak * abs(volume_t - volumes[-2])
        
        # Volume-Momentum Alignment
        daily_alignment = np.sign(mom_1d) * np.sign(volume_t - volumes[-2])
        
        # Alignment persistence
        alignment_persistence = 1
        for j in range(len(recent_mom_signs)-1, 0, -1):
            alignment_j = recent_mom_signs[j] * volume_signs[j]
            alignment_j_minus = recent_mom_signs[j-1] * volume_signs[j-1]
            if alignment_j > 0 and alignment_j_minus > 0:
                alignment_persistence += 1
            else:
                break
        
        # Alignment strength
        alignment_strength = alignment_persistence * abs(mom_1d)
        
        # Volume Regime Persistence
        volume_ratio_3d = np.sum(volumes[-3:]) / np.sum(volumes)
        volume_regime = "normal"
        if volume_ratio_3d > 1.1:
            volume_regime = "high"
        elif volume_ratio_3d < 0.9:
            volume_regime = "low"
        
        # Regime persistence
        regime_persistence = 1
        for j in range(i-1, max(i-10, 0), -1):
            prev_volumes = data['volume'].iloc[j-2:j+1].values if j >= 2 else data['volume'].iloc[:j+1].values
            prev_total_volumes = data['volume'].iloc[max(0, j-9):j+1].values
            prev_ratio = np.sum(prev_volumes) / np.sum(prev_total_volumes) if len(prev_total_volumes) > 0 else 1
            
            prev_regime = "normal"
            if prev_ratio > 1.1:
                prev_regime = "high"
            elif prev_ratio < 0.9:
                prev_regime = "low"
            
            if prev_regime == volume_regime:
                regime_persistence += 1
            else:
                break
        
        # ADAPTIVE REGIME DETECTION
        # Volatility Regime
        volatility_ratio = np.sum(daily_ranges[-3:]) / range_volatility_10d if range_volatility_10d != 0 else 1
        volatility_regime = "normal"
        if volatility_ratio > 1.15:
            volatility_regime = "high"
        elif volatility_ratio < 0.85:
            volatility_regime = "low"
        
        # Trend Regime
        trend_strength = abs(mom_10d) / range_volatility_10d if range_volatility_10d != 0 else 0
        trend_regime = "moderate"
        if trend_strength > 0.6:
            trend_regime = "strong"
        elif trend_strength < 0.3:
            trend_regime = "weak"
        
        # Persistence Regime
        combined_persistence = min(direction_streak, volume_streak, alignment_persistence)
        
        # ADAPTIVE FACTOR CONSTRUCTION
        # Core Momentum Signal
        core_momentum = (3 * vam_3d + 1 * vam_10d) / 4
        persistence_weighted = core_momentum * (1 + combined_persistence / 8)
        acceleration_adjusted = persistence_weighted * (1 + 0.1 * np.sign(mom_acceleration))
        
        # Volume-Price Confirmation
        alignment_multiplier = 1 + (alignment_persistence / 6)
        volume_persistence_multiplier = 1 + (volume_streak / 10)
        combined_confirmation = alignment_multiplier * volume_persistence_multiplier
        
        # Regime-Adaptive Scaling
        # Volatility scaling
        volatility_scaling = 1.0
        if volatility_regime == "high":
            volatility_scaling = 0.7
        elif volatility_regime == "low":
            volatility_scaling = 1.3
        
        # Volume regime scaling
        volume_scaling = 1.0
        if volume_regime == "high":
            volume_scaling = 1.2
        elif volume_regime == "low":
            volume_scaling = 0.8
        
        # Trend regime scaling
        trend_scaling = 1.0
        if trend_regime == "strong":
            trend_scaling = 1.4
        elif trend_regime == "weak":
            trend_scaling = 0.6
        
        # Persistence Quality Bonus
        quality_multiplier = 1 + (combined_persistence / 10)
        
        # FINAL ALPHA CALCULATION
        base_signal = acceleration_adjusted
        volume_enhanced = base_signal * combined_confirmation
        regime_adjusted = volume_enhanced * volatility_scaling * volume_scaling * trend_scaling
        final_alpha = regime_adjusted * quality_multiplier
        
        alpha.loc[current_date] = final_alpha
    
    # Fill NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
