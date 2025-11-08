import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate price changes
    data['close_change'] = data['close'].diff()
    
    # 1. Asymmetric Price Fractal Dimension
    # Upside and downside path lengths over 5 days
    upside_path = pd.Series(index=data.index, dtype=float)
    downside_path = pd.Series(index=data.index, dtype=float)
    
    for i in range(5, len(data)):
        # Calculate upside path length (sum of positive changes)
        upside_sum = 0
        downside_sum = 0
        for j in range(i-4, i+1):
            if j > 0:
                change = data['close'].iloc[j] - data['close'].iloc[j-1]
                upside_sum += max(0, change)
                downside_sum += abs(min(0, change))
        upside_path.iloc[i] = upside_sum
        downside_path.iloc[i] = downside_sum
    
    # Asymmetric price ranges over 3 days
    upside_range = pd.Series(index=data.index, dtype=float)
    downside_range = pd.Series(index=data.index, dtype=float)
    
    for i in range(3, len(data)):
        upside_sum = 0
        downside_sum = 0
        for j in range(i-2, i+1):
            if j > 0:
                upside_sum += data['high'].iloc[j] - data['close'].iloc[j-1]
                downside_sum += data['close'].iloc[j-1] - data['low'].iloc[j]
        upside_range.iloc[i] = upside_sum / 3
        downside_range.iloc[i] = downside_sum / 3
    
    # Compute asymmetric fractal dimensions
    upside_fd = 1 + np.log(upside_path + 1e-8) / np.log(upside_range + 1e-8)
    downside_fd = 1 + np.log(downside_path + 1e-8) / np.log(downside_range + 1e-8)
    fd_asymmetry = upside_fd - downside_fd
    
    # 2. Volume Fractal Regime Indicators
    # Volume burst detection (3-day acceleration)
    volume_acceleration = (data['volume'] - data['volume'].shift(2)) / (data['volume'].shift(2) + 1)
    
    # Volume contraction detection (5-day decay)
    volume_decay = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        current_vol = data['volume'].iloc[i]
        max_past_vol = max(data['volume'].iloc[i-5:i])
        volume_decay.iloc[i] = current_vol / (max_past_vol + 1e-8)
    
    # Volume path smoothness (2-day vs 5-day path length)
    vol_path_2day = data['volume'].diff().abs().rolling(window=2).sum()
    vol_path_5day = data['volume'].diff().abs().rolling(window=5).sum()
    volume_smoothness = vol_path_2day / (vol_path_5day + 1e-8)
    
    # Volume range consistency (2-day vs 5-day range sums)
    vol_range_2day = (data['volume'].rolling(window=2).max() - data['volume'].rolling(window=2).min())
    vol_range_5day = (data['volume'].rolling(window=5).max() - data['volume'].rolling(window=5).min())
    volume_consistency = vol_range_2day / (vol_range_5day + 1e-8)
    
    # Volume regime classification
    high_vol_threshold = 0.3
    low_vol_threshold = 0.7
    volume_regime = pd.Series(index=data.index, dtype=int)  # 0: low, 1: transition, 2: high
    
    for i in range(len(data)):
        if i >= 5:
            if volume_acceleration.iloc[i] > high_vol_threshold:
                volume_regime.iloc[i] = 2  # High volatility
            elif volume_decay.iloc[i] < low_vol_threshold:
                volume_regime.iloc[i] = 0  # Low volatility
            else:
                volume_regime.iloc[i] = 1  # Transition
    
    # 3. Cross-Asymmetry Patterns
    # Price Upside FD vs Volume Regime Interaction
    upside_fd_vol_interaction = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            if volume_regime.iloc[i] == 0:  # Volume contraction
                upside_fd_vol_interaction.iloc[i] = upside_fd.iloc[i]
            elif volume_regime.iloc[i] == 2:  # Volume burst
                upside_fd_vol_interaction.iloc[i] = -upside_fd.iloc[i]
            else:
                upside_fd_vol_interaction.iloc[i] = 0
    
    # Price Downside FD vs Volume Pattern Divergence
    downside_fd_vol_divergence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            if volume_regime.iloc[i] == 1:  # Stable volume (transition regime)
                downside_fd_vol_divergence.iloc[i] = downside_fd.iloc[i]
            elif volume_regime.iloc[i] == 2:  # Volatile volume
                downside_fd_vol_divergence.iloc[i] = -downside_fd.iloc[i]
            else:
                downside_fd_vol_divergence.iloc[i] = 0
    
    # Asymmetry momentum
    fd_asymmetry_change = fd_asymmetry.diff()
    volume_regime_change = volume_regime.diff()
    asymmetry_momentum = fd_asymmetry_change * volume_regime_change
    
    # 4. Multi-Scale Fractal Factors
    # 3-day FD asymmetry
    fd_asymmetry_3day = pd.Series(index=data.index, dtype=float)
    for i in range(3, len(data)):
        upside_sum_3day = 0
        downside_sum_3day = 0
        upside_range_3day = 0
        downside_range_3day = 0
        
        for j in range(i-2, i+1):
            if j > 0:
                # Path lengths
                change = data['close'].iloc[j] - data['close'].iloc[j-1]
                upside_sum_3day += max(0, change)
                downside_sum_3day += abs(min(0, change))
                
                # Ranges
                upside_range_3day += data['high'].iloc[j] - data['close'].iloc[j-1]
                downside_range_3day += data['close'].iloc[j-1] - data['low'].iloc[j]
        
        upside_fd_3day = 1 + np.log(upside_sum_3day + 1e-8) / np.log((upside_range_3day/3) + 1e-8)
        downside_fd_3day = 1 + np.log(downside_sum_3day + 1e-8) / np.log((downside_range_3day/3) + 1e-8)
        fd_asymmetry_3day.iloc[i] = upside_fd_3day - downside_fd_3day
    
    # Short-Long Fractal Asymmetry Ratio
    short_long_ratio = fd_asymmetry_3day / (fd_asymmetry + 1e-8)
    
    # Fractal Regime Transition Strength
    regime_transition_strength = pd.Series(index=data.index, dtype=float)
    for i in range(6, len(data)):
        if volume_regime.iloc[i] != volume_regime.iloc[i-1]:
            regime_transition_strength.iloc[i] = abs(fd_asymmetry.iloc[i] - fd_asymmetry.iloc[i-1])
        else:
            regime_transition_strength.iloc[i] = 0
    
    # Cross-Timeframe Pattern Alignment
    timeframe_alignment = (fd_asymmetry_3day * fd_asymmetry).fillna(0)
    
    # 5. Generate Regime-Aware Predictive Signals
    final_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(6, len(data)):
        # Base components
        base_signal = (
            upside_fd_vol_interaction.iloc[i] +
            downside_fd_vol_divergence.iloc[i] +
            asymmetry_momentum.iloc[i] * 0.5
        )
        
        # Multi-scale enhancement
        multi_scale_enhancement = (
            short_long_ratio.iloc[i] * 0.3 +
            regime_transition_strength.iloc[i] * 0.4 +
            timeframe_alignment.iloc[i] * 0.3
        )
        
        # Volume regime weighting
        if volume_regime.iloc[i] == 0:  # Low volatility - emphasize stability
            regime_weight = volume_smoothness.iloc[i] * volume_consistency.iloc[i]
        elif volume_regime.iloc[i] == 2:  # High volatility - emphasize transitions
            regime_weight = regime_transition_strength.iloc[i]
        else:  # Transition regime - balance both
            regime_weight = 0.5
        
        final_factor.iloc[i] = base_signal * (1 + multi_scale_enhancement) * (1 + regime_weight * 0.2)
    
    return final_factor.fillna(0)
