import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Fractal Price-Volume Asymmetry Framework
    # Intraday Fractal Patterns
    morning_fractal = (df['high'] - df['open']) / (df['open'] - df['low'] + 0.001)
    afternoon_fractal = (df['close'] - df['low']) / (df['high'] - df['close'] + 0.001)
    fractal_shift = morning_fractal - afternoon_fractal
    
    high_low_range = df['high'] - df['low'] + 0.001
    opening_fractal_strength = (df['open'] - df['low']) / high_low_range
    closing_fractal_strength = (df['high'] - df['close']) / high_low_range
    
    # Volume Fractal Distribution
    morning_volume_fractal = df['volume'] * opening_fractal_strength
    afternoon_volume_fractal = df['volume'] * closing_fractal_strength
    volume_fractal_divergence = morning_volume_fractal - afternoon_volume_fractal
    
    # Fractal Asymmetry Integration
    price_fractal_asymmetry = fractal_shift * (opening_fractal_strength - closing_fractal_strength)
    volume_fractal_asymmetry = volume_fractal_divergence * fractal_shift
    combined_fractal_asymmetry = price_fractal_asymmetry * volume_fractal_asymmetry
    
    # Multi-Timeframe Momentum & Persistence
    # Fractal Momentum Framework
    short_term_fractal_momentum = fractal_shift - fractal_shift.shift(3)
    medium_term_fractal_momentum = fractal_shift - fractal_shift.shift(8)
    fractal_momentum_convergence = np.sign(short_term_fractal_momentum) * np.sign(medium_term_fractal_momentum)
    
    # Volume Fractal Persistence
    volume_fractal_persistence = pd.Series(index=df.index, dtype=float)
    volume_fractal_reversal = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i == 0:
            volume_fractal_persistence.iloc[i] = 1 if volume_fractal_divergence.iloc[i] > 0 else 0
            volume_fractal_reversal.iloc[i] = 0
        else:
            if volume_fractal_divergence.iloc[i] > 0:
                if volume_fractal_divergence.iloc[i-1] > 0:
                    volume_fractal_persistence.iloc[i] = volume_fractal_persistence.iloc[i-1] + 1
                else:
                    volume_fractal_persistence.iloc[i] = 1
            else:
                volume_fractal_persistence.iloc[i] = 0
            
            if i >= 1:
                if (volume_fractal_divergence.iloc[i] > 0 and volume_fractal_divergence.iloc[i-1] <= 0) or \
                   (volume_fractal_divergence.iloc[i] <= 0 and volume_fractal_divergence.iloc[i-1] > 0):
                    volume_fractal_reversal.iloc[i] = volume_fractal_reversal.iloc[i-1] + 1
                else:
                    volume_fractal_reversal.iloc[i] = volume_fractal_reversal.iloc[i-1]
    
    volume_fractal_stability = volume_fractal_persistence / (volume_fractal_reversal + 1)
    
    # Fractal Pattern Persistence
    fractal_consistency = pd.Series(index=df.index, dtype=float)
    fractal_sign_changes = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i == 0:
            fractal_consistency.iloc[i] = 1
            fractal_sign_changes.iloc[i] = 0
        else:
            if np.sign(fractal_shift.iloc[i]) == np.sign(fractal_shift.iloc[i-1]):
                fractal_consistency.iloc[i] = fractal_consistency.iloc[i-1] + 1
            else:
                fractal_consistency.iloc[i] = 1
            
            if i >= 1:
                if np.sign(fractal_shift.iloc[i]) != np.sign(fractal_shift.iloc[i-1]):
                    fractal_sign_changes.iloc[i] = fractal_sign_changes.iloc[i-1] + 1
                else:
                    fractal_sign_changes.iloc[i] = fractal_sign_changes.iloc[i-1]
    
    fractal_stability_ratio = fractal_consistency / (fractal_sign_changes + 1)
    fractal_pattern_quality = fractal_consistency * fractal_stability_ratio
    
    # Convergence & Divergence Detection
    # Price-Volume Fractal Alignment
    fractal_convergence_signal = np.sign(fractal_shift) * np.sign(volume_fractal_divergence)
    fractal_convergence_strength = np.abs(fractal_shift * volume_fractal_divergence)
    converged_fractal_micro = fractal_convergence_signal * fractal_convergence_strength
    
    # Momentum Divergence Analysis
    momentum_fractal_divergence = fractal_momentum_convergence * volume_fractal_divergence
    persistence_divergence = fractal_pattern_quality - volume_fractal_stability
    combined_divergence = momentum_fractal_divergence * persistence_divergence
    
    # Asymmetric Regime Framework
    # Volatility Regime Detection
    current_range = df['high'] - df['low'] + 0.001
    past_range = (df['high'].shift(5) - df['low'].shift(5) + 0.001)
    volatility_ratio = current_range / past_range
    
    regime_multiplier = pd.Series(1.0, index=df.index)
    regime_multiplier = np.where(volatility_ratio > 1.2, 1.5, 
                                np.where(past_range / current_range > 1.2, 0.7, 1.0))
    
    # Efficiency-Based Filtering
    absolute_efficiency = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
    efficiency_multiplier = np.where(absolute_efficiency > 0.6, 2.0,
                                   np.where(absolute_efficiency < 0.3, 0.5, 1.0))
    
    # Alpha Component Integration
    # Core Fractal Signal
    base_fractal_signal = combined_fractal_asymmetry * converged_fractal_micro
    momentum_enhanced_signal = base_fractal_signal * fractal_momentum_convergence
    
    # Quality & Persistence Enhancement
    quality_score = fractal_pattern_quality * volume_fractal_stability
    persistence_enhanced_signal = momentum_enhanced_signal * quality_score
    divergence_enhanced_signal = persistence_enhanced_signal * combined_divergence
    
    # Final Alpha Construction
    # Regime-Adaptive Scaling
    volatility_scaled_signal = divergence_enhanced_signal * regime_multiplier
    efficiency_scaled_signal = volatility_scaled_signal * efficiency_multiplier
    
    # Final Alpha
    final_alpha = efficiency_scaled_signal * (df['close'] - df['open']) * np.log(df['volume'] + 1)
    
    return final_alpha
