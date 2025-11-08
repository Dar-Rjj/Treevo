import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Absolute Momentum with Volume Confirmation alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate required lookback periods
    for i in range(len(df)):
        if i < 10:  # Need at least 10 days of data
            alpha.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        prev_data = df.iloc[max(0, i-1):i+1]
        
        # Raw Price Components
        intraday_momentum = current_data['close'] - current_data['open']
        daily_range = current_data['high'] - current_data['low']
        volume_t = current_data['volume']
        
        # Absolute Momentum Framework
        # Very Short-Term (1-3 days)
        momentum_1d = intraday_momentum
        
        momentum_2d = ((current_data['close'] - df.iloc[i-1]['close']) + 
                      (df.iloc[i-1]['close'] - df.iloc[i-2]['close']))
        
        momentum_3d = sum(df.iloc[j]['close'] - df.iloc[j]['open'] for j in range(i-2, i+1))
        
        # Short-Term (5 days)
        momentum_5d = sum(df.iloc[j]['close'] - df.iloc[j]['open'] for j in range(i-4, i+1))
        
        # Medium-Term (10 days)
        momentum_10d = sum(df.iloc[j]['close'] - df.iloc[j]['open'] for j in range(i-9, i+1))
        
        # Volume Regime Analysis
        # Volume Trend Components
        volume_change_1d = volume_t - df.iloc[i-1]['volume']
        volume_change_5d = volume_t - df.iloc[i-4]['volume']
        
        volume_trend_strength = sum(
            np.sign(df.iloc[j]['volume'] - df.iloc[j-1]['volume']) 
            for j in range(i-4, i+1) if j > 0
        )
        
        # Volume Regime Classification
        volume_avg_prev_4d = np.mean([df.iloc[j]['volume'] for j in range(i-4, i)])
        volume_ratio_5d = volume_t / volume_avg_prev_4d if volume_avg_prev_4d > 0 else 1.0
        
        if volume_ratio_5d > 1.15:
            volume_regime = 'high'
        elif volume_ratio_5d >= 0.85:
            volume_regime = 'normal'
        else:
            volume_regime = 'low'
        
        # Volume-Momentum Confirmation
        direction_alignment = np.sign(intraday_momentum) * np.sign(volume_change_1d)
        
        alignment_consistency = sum(
            np.sign(df.iloc[j]['close'] - df.iloc[j]['open']) * 
            np.sign(df.iloc[j]['volume'] - df.iloc[j-1]['volume']) > 0
            for j in range(i-4, i+1) if j > 0
        )
        
        volume_confidence_score = alignment_consistency * abs(volume_trend_strength)
        
        # Volatility Context
        # Volatility Measurement
        range_5d = np.mean([df.iloc[j]['high'] - df.iloc[j]['low'] for j in range(i-4, i+1)])
        range_10d = np.mean([df.iloc[j]['high'] - df.iloc[j]['low'] for j in range(i-9, i+1)])
        
        # Volatility Regime
        volatility_ratio = range_5d / range_10d if range_10d > 0 else 1.0
        
        if volatility_ratio > 1.25:
            volatility_regime = 'high'
        elif volatility_ratio >= 0.75:
            volatility_regime = 'normal'
        else:
            volatility_regime = 'low'
        
        # Regime-Adaptive Factor Construction
        # Base Momentum Signal with Regime-Based Weighting
        if volatility_regime == 'high':
            base_momentum = (6 * momentum_1d + 3 * momentum_5d + 1 * momentum_10d) / 10
        elif volatility_regime == 'normal':
            base_momentum = (4 * momentum_1d + 3 * momentum_5d + 3 * momentum_10d) / 10
        else:  # low volatility
            base_momentum = (2 * momentum_1d + 3 * momentum_5d + 5 * momentum_10d) / 10
        
        momentum_acceleration = momentum_5d - momentum_10d
        
        # Volume Confirmation Multiplier
        if volume_regime == 'high':
            volume_regime_effect = 1.3
        elif volume_regime == 'normal':
            volume_regime_effect = 1.0
        else:  # low volume
            volume_regime_effect = 0.7
        
        alignment_boost = 1 + (volume_confidence_score / 10)
        
        # Volatility Scaling
        if volatility_regime == 'high':
            volatility_scaling = 0.6
        elif volatility_regime == 'normal':
            volatility_scaling = 1.0
        else:  # low volatility
            volatility_scaling = 1.4
        
        # Final Alpha Output
        alpha_value = base_momentum * volume_regime_effect * alignment_boost * volatility_scaling
        momentum_enhancement = alpha_value * (1 + 0.1 * np.sign(momentum_acceleration))
        
        alpha.iloc[i] = momentum_enhancement
    
    # Handle any remaining NaN values
    alpha = alpha.fillna(0)
    
    return alpha
