import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        # Multi-Scale Fractal Analysis
        # Short-Term Hurst (5-day)
        if i >= 4:
            close_5 = df['close'].iloc[i-4:i+1]
            range_5 = close_5.max() - close_5.min()
            std_5 = close_5.std()
            if std_5 > 0:
                hurst_short = np.log(range_5 / std_5) / np.log(5)
            else:
                hurst_short = 0
        else:
            hurst_short = 0
            
        # Medium-Term Hurst (20-day)
        close_20 = df['close'].iloc[i-19:i+1]
        range_20 = close_20.max() - close_20.min()
        std_20 = close_20.std()
        if std_20 > 0:
            hurst_medium = np.log(range_20 / std_20) / np.log(20)
        else:
            hurst_medium = 0
            
        fractal_dim_change = hurst_medium - hurst_short
        
        # Entropy-Enhanced Momentum
        if i >= 2:
            quad_momentum = (df['close'].iloc[i] / df['close'].iloc[i-2])**2 - (df['close'].iloc[i] / df['close'].iloc[i-1])**2
            momentum_curvature = (df['close'].iloc[i] - 2*df['close'].iloc[i-1] + df['close'].iloc[i-2]) / df['close'].iloc[i-2]
            entropy_weight = (df['high'].iloc[i] - df['low'].iloc[i]) / (df['close'].iloc[i] - df['open'].iloc[i] + 1e-8)
            entropy_weighted_momentum = quad_momentum * entropy_weight
        else:
            quad_momentum = 0
            momentum_curvature = 0
            entropy_weighted_momentum = 0
            
        # Fractal-Momentum Synchronization
        fractal_enhanced_momentum = entropy_weighted_momentum * fractal_dim_change
        
        if i >= 1:
            vol_ratio = df['volume'].iloc[i] / df['volume'].iloc[i-1]
            price_range_ratio = (df['high'].iloc[i] - df['low'].iloc[i]) / (df['high'].iloc[i-1] - df['low'].iloc[i-1] + 1e-8)
            if price_range_ratio > 0:
                volume_fractal_dim = np.log(vol_ratio) / np.log(price_range_ratio)
            else:
                volume_fractal_dim = 0
        else:
            volume_fractal_dim = 0
            
        fractal_volume_alignment = fractal_enhanced_momentum * volume_fractal_dim
        
        # Pressure-Entropy Asymmetry
        up_pressure = (df['high'].iloc[i] - df['open'].iloc[i]) * df['volume'].iloc[i]
        down_pressure = (df['open'].iloc[i] - df['low'].iloc[i]) * df['volume'].iloc[i]
        total_pressure = up_pressure + down_pressure
        if total_pressure > 0:
            net_pressure = (up_pressure - down_pressure) / total_pressure
        else:
            net_pressure = 0
            
        # Volume-Price Entropy
        if i >= 2:
            volume_entropy = df['volume'].iloc[i] / (df['volume'].iloc[i-1] + df['volume'].iloc[i-2] + 1e-8)
        else:
            volume_entropy = 0
            
        # Price-Volume Entropy (5-day window)
        if i >= 4:
            prices = df['close'].iloc[i-4:i+1]
            volumes = df['volume'].iloc[i-4:i+1]
            avg_price = prices.mean()
            total_vol = volumes.sum()
            if total_vol > 0:
                price_vol_entropy = -((prices - avg_price)**2 * volumes).sum() / total_vol
            else:
                price_vol_entropy = 0
        else:
            price_vol_entropy = 0
            
        # Volume-Weighted Price Dispersion
        if df['amount'].iloc[i] > 0:
            vol_weighted_dispersion = (df['high'].iloc[i] - df['low'].iloc[i]) * df['volume'].iloc[i] / df['amount'].iloc[i]
        else:
            vol_weighted_dispersion = 0
            
        # Pressure-Entropy Coupling
        entropy_weighted_pressure = net_pressure * price_vol_entropy
        if down_pressure > 0:
            pressure_efficiency_asymmetry = (up_pressure / (df['high'].iloc[i] - df['low'].iloc[i])) / (down_pressure / (df['high'].iloc[i] - df['low'].iloc[i]))
        else:
            pressure_efficiency_asymmetry = 0
        volume_pressure_entropy = entropy_weighted_pressure * volume_entropy
        
        # Fracture-Regime Detection
        if i >= 1:
            fracture_intensity = abs(df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i-1] - df['low'].iloc[i-1] + 1e-8)
        else:
            fracture_intensity = 0
            
        if i >= 2:
            multi_period_fracture = np.sign(df['close'].iloc[i] - df['close'].iloc[i-1]) * np.sign(df['close'].iloc[i-1] - df['close'].iloc[i-2])
        else:
            multi_period_fracture = 0
            
        fracture_entropy = fracture_intensity * (df['high'].iloc[i] - df['low'].iloc[i]) / (df['close'].iloc[i] - df['open'].iloc[i] + 1e-8)
        
        # Regime Classification
        high_fracture_regime = 1 if fracture_intensity > volume_entropy else 0
        low_fracture_regime = 1 if fracture_intensity < volume_entropy * 0.5 else 0
        if i >= 1:
            amount_ratio = abs(df['amount'].iloc[i] / df['amount'].iloc[i-1] - 1)
        else:
            amount_ratio = 0
        transition_regime = 1 if abs(fracture_intensity - volume_entropy) > amount_ratio else 0
        
        # Fracture-Pressure Dynamics
        fracture_enhanced_pressure = net_pressure * fracture_intensity
        regime_pressure_alignment = fracture_enhanced_pressure * high_fracture_regime
        transition_pressure = net_pressure * transition_regime
        
        # Acceleration Breakout Detection
        if i >= 1:
            asymmetric_momentum = (df['high'].iloc[i] - df['close'].iloc[i-1]) / (df['close'].iloc[i-1] - df['low'].iloc[i] + 1e-8)
        else:
            asymmetric_momentum = 0
            
        if i >= 3:
            prev_highs = [df['high'].iloc[i-1], df['high'].iloc[i-2], df['high'].iloc[i-3]]
            breakout_momentum = 1 if df['close'].iloc[i] > max(prev_highs) else 0
            breakout_momentum *= quad_momentum
        else:
            breakout_momentum = 0
            
        accelerated_fractal = breakout_momentum * fractal_dim_change
        
        # Volume-Pressure Surge
        if i >= 3:
            avg_vol = (df['volume'].iloc[i-1] + df['volume'].iloc[i-2] + df['volume'].iloc[i-3]) / 3
            volume_surge = 1 if df['volume'].iloc[i] > 1.5 * avg_vol else 0
        else:
            volume_surge = 0
            
        if i >= 1:
            pressure_surge = 1 if abs(net_pressure) > abs(net_pressure) else 0
        else:
            pressure_surge = 0
            
        surge_entropy = volume_surge * pressure_surge * volume_entropy
        
        # Acceleration-Entropy Synchronization
        momentum_entropy_acceleration = accelerated_fractal * price_vol_entropy
        pressure_enhanced_acceleration = momentum_entropy_acceleration * net_pressure
        fracture_aligned_acceleration = pressure_enhanced_acceleration * fracture_intensity
        
        # Regime-Adaptive Signal Integration
        high_fracture_momentum = fractal_enhanced_momentum * high_fracture_regime
        high_pressure_entropy = volume_pressure_entropy * high_fracture_regime
        high_acceleration = fracture_aligned_acceleration * high_fracture_regime
        
        low_fracture_momentum = momentum_curvature * low_fracture_regime
        low_pressure_stability = net_pressure * low_fracture_regime
        low_volume_alignment = volume_fractal_dim * low_fracture_regime
        
        transition_momentum = pressure_enhanced_acceleration * transition_regime
        transition_entropy = entropy_weighted_pressure * transition_regime
        transition_pressure_vol = transition_pressure * volume_entropy
        
        # Multi-Scale Factor Synthesis
        fractal_pressure_core = fractal_volume_alignment * volume_pressure_entropy
        entropy_acceleration_core = momentum_entropy_acceleration * price_vol_entropy
        fracture_regime_core = fracture_enhanced_pressure * regime_pressure_alignment
        
        high_regime_signal = high_fracture_momentum * high_pressure_entropy * high_acceleration
        low_regime_signal = low_fracture_momentum * low_pressure_stability * low_volume_alignment
        transition_signal = transition_momentum * transition_entropy * transition_pressure_vol
        
        fractal_confirmed_signal = (high_regime_signal + low_regime_signal + transition_signal) * fractal_dim_change
        
        if df['volume'].iloc[i] > 0:
            volume_distribution_adjustment = fractal_confirmed_signal / (df['amount'].iloc[i] / df['volume'].iloc[i])
        else:
            volume_distribution_adjustment = 0
            
        pressure_asymmetry_final = volume_distribution_adjustment * pressure_efficiency_asymmetry
        
        # Final Alpha Generation
        primary_factor = pressure_asymmetry_final * vol_weighted_dispersion
        acceleration_confirmation = primary_factor * asymmetric_momentum
        final_alpha = acceleration_confirmation * surge_entropy
        
        result.iloc[i] = final_alpha
    
    return result
