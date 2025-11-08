import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Volatility-Scaled Momentum with Volume-Price Persistence alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate required lookback periods
    for i in range(len(data)):
        if i < 20:  # Need at least 20 days of history
            alpha.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # Extract current day prices
        open_t = current_data['open'].iloc[-1]
        high_t = current_data['high'].iloc[-1]
        low_t = current_data['low'].iloc[-1]
        close_t = current_data['close'].iloc[-1]
        volume_t = current_data['volume'].iloc[-1]
        
        # Extract historical prices and volumes
        close_hist = current_data['close'].iloc[-21:-1].values  # t-20 to t-1
        high_hist = current_data['high'].iloc[-21:-1].values
        low_hist = current_data['low'].iloc[-21:-1].values
        volume_hist = current_data['volume'].iloc[-21:-1].values
        
        # Volatility-Scaled Momentum Framework
        # Multi-Timeframe Price Momentum
        momentum_1d = close_t - close_hist[-1]  # Close_t - Close_{t-1}
        momentum_3d = close_t - close_hist[-3]  # Close_t - Close_{t-3}
        momentum_5d = close_t - close_hist[-5]  # Close_t - Close_{t-5}
        momentum_10d = close_t - close_hist[-10]  # Close_t - Close_{t-10}
        
        # Range-Based Volatility
        daily_range = high_t - low_t
        short_term_vol = np.sum(high_hist[-3:] - low_hist[-3:]) / 3  # t-2 to t
        medium_term_vol = np.sum(high_hist[-5:] - low_hist[-5:]) / 5  # t-4 to t
        long_term_vol = np.sum(high_hist[-10:] - low_hist[-10:]) / 10  # t-9 to t
        
        # Volatility-Scaled Momentum Signals
        vsm_1d = momentum_1d / daily_range if daily_range != 0 else 0
        vsm_3d = momentum_3d / short_term_vol if short_term_vol != 0 else 0
        vsm_5d = momentum_5d / medium_term_vol if medium_term_vol != 0 else 0
        vsm_10d = momentum_10d / long_term_vol if long_term_vol != 0 else 0
        
        # Volume-Price Persistence Analysis
        # Volume Trend Components
        volume_change = volume_t - volume_hist[-1]
        volume_direction = np.sign(volume_change)
        
        # Calculate volume direction persistence
        volume_dir_persistence = 1
        for j in range(1, min(10, len(volume_hist))):
            if np.sign(volume_hist[-j] - volume_hist[-j-1]) == volume_direction:
                volume_dir_persistence += 1
            else:
                break
        
        volume_momentum = volume_change / volume_hist[-1] if volume_hist[-1] != 0 else 0
        
        # Price-Volume Alignment
        price_change = close_t - close_hist[-1]
        alignment_signal = np.sign(price_change) * volume_direction
        
        # Calculate alignment persistence
        alignment_persistence = 1
        for j in range(1, min(8, len(close_hist))):
            price_chg_hist = close_hist[-j] - close_hist[-j-1]
            vol_chg_hist = volume_hist[-j] - volume_hist[-j-1]
            if np.sign(price_chg_hist) * np.sign(vol_chg_hist) > 0:
                alignment_persistence += 1
            else:
                break
        
        alignment_strength = alignment_persistence * abs(price_change) / daily_range if daily_range != 0 else 0
        
        # Cumulative alignment (last 5 days)
        cumulative_alignment = 0
        for j in range(5):
            if i-j-1 >= 0:
                price_chg = close_hist[-j-1] - (close_hist[-j-2] if j < 4 else close_hist[-j-2])
                vol_chg = volume_hist[-j-1] - (volume_hist[-j-2] if j < 4 else volume_hist[-j-2])
                cumulative_alignment += np.sign(price_chg) * np.sign(vol_chg)
        
        # Volume Regime Classification
        recent_volume = np.mean(volume_hist[-3:])
        baseline_volume = np.mean(volume_hist[-10:-3])
        volume_ratio = recent_volume / baseline_volume if baseline_volume != 0 else 1
        
        if volume_ratio > 1.2:
            volume_regime = 'high'
        elif volume_ratio >= 0.8:
            volume_regime = 'normal'
        else:
            volume_regime = 'low'
        
        # Adaptive Timeframe Weighting
        # Volatility Regime Detection
        volatility_ratio = short_term_vol / long_term_vol if long_term_vol != 0 else 1
        
        if volatility_ratio > 1.3:
            volatility_regime = 'high'
            weights = {'ultra_short': 0.6, 'short_term': 0.3, 'medium_term': 0.1, 'long_term': 0.0}
        elif volatility_ratio >= 0.7:
            volatility_regime = 'normal'
            weights = {'ultra_short': 0.3, 'short_term': 0.4, 'medium_term': 0.2, 'long_term': 0.1}
        else:
            volatility_regime = 'low'
            weights = {'ultra_short': 0.1, 'short_term': 0.3, 'medium_term': 0.4, 'long_term': 0.2}
        
        # Weighted Momentum Composite
        weighted_vsm = (vsm_1d * weights['ultra_short'] + 
                       vsm_3d * weights['short_term'] + 
                       vsm_5d * weights['medium_term'] + 
                       vsm_10d * weights['long_term'])
        
        # Volume Confidence
        volume_confidence = 1 + (volume_dir_persistence / 10)
        volume_adjusted_momentum = weighted_vsm * volume_confidence
        
        # Persistence Enhancement Framework
        # Momentum Acceleration
        short_term_acceleration = vsm_3d - vsm_10d
        medium_term_acceleration = vsm_5d - vsm_10d
        acceleration_direction = np.sign(short_term_acceleration)
        
        # Alignment Persistence Boost
        persistence_multiplier = 1 + (alignment_persistence / 8)
        
        # Volume Regime Confidence
        if volume_regime == 'high':
            volume_regime_multiplier = 1.3
        elif volume_regime == 'normal':
            volume_regime_multiplier = 1.0
        else:
            volume_regime_multiplier = 0.7
        
        # Final Alpha Construction
        # Core Momentum Component
        core_momentum = volume_adjusted_momentum * persistence_multiplier * volume_regime_multiplier
        
        # Acceleration Adjustment
        if acceleration_direction > 0:
            acceleration_factor = 1 + 0.15 * acceleration_direction
        elif acceleration_direction < 0:
            acceleration_factor = 1 - 0.15 * abs(acceleration_direction)
        else:
            acceleration_factor = 1.0
        
        # Cumulative Alignment Enhancement
        if cumulative_alignment >= 3:
            alignment_factor = 1.2
        elif cumulative_alignment <= -3:
            alignment_factor = 0.8
        else:
            alignment_factor = 1.0
        
        # Final Alpha Value
        final_alpha = core_momentum * acceleration_factor * alignment_factor
        
        alpha.iloc[i] = final_alpha
    
    return alpha
