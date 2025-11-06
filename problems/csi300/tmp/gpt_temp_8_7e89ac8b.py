import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Avoid division by zero
    epsilon = 1e-8
    
    for i in range(8, len(df)):
        # Current data
        open_t = df['open'].iloc[i]
        high_t = df['high'].iloc[i]
        low_t = df['low'].iloc[i]
        close_t = df['close'].iloc[i]
        amount_t = df['amount'].iloc[i]
        volume_t = df['volume'].iloc[i]
        
        # Historical data
        close_t_1 = df['close'].iloc[i-1] if i >= 1 else np.nan
        close_t_2 = df['close'].iloc[i-2] if i >= 2 else np.nan
        close_t_3 = df['close'].iloc[i-3] if i >= 3 else np.nan
        close_t_4 = df['close'].iloc[i-4] if i >= 4 else np.nan
        close_t_5 = df['close'].iloc[i-5] if i >= 5 else np.nan
        close_t_8 = df['close'].iloc[i-8] if i >= 8 else np.nan
        
        high_t_1 = df['high'].iloc[i-1] if i >= 1 else np.nan
        high_t_2 = df['high'].iloc[i-2] if i >= 2 else np.nan
        high_t_3 = df['high'].iloc[i-3] if i >= 3 else np.nan
        high_t_4 = df['high'].iloc[i-4] if i >= 4 else np.nan
        high_t_8 = df['high'].iloc[i-8] if i >= 8 else np.nan
        
        low_t_1 = df['low'].iloc[i-1] if i >= 1 else np.nan
        low_t_2 = df['low'].iloc[i-2] if i >= 2 else np.nan
        low_t_3 = df['low'].iloc[i-3] if i >= 3 else np.nan
        low_t_4 = df['low'].iloc[i-4] if i >= 4 else np.nan
        low_t_8 = df['low'].iloc[i-8] if i >= 8 else np.nan
        
        volume_t_1 = df['volume'].iloc[i-1] if i >= 1 else np.nan
        volume_t_2 = df['volume'].iloc[i-2] if i >= 2 else np.nan
        volume_t_3 = df['volume'].iloc[i-3] if i >= 3 else np.nan
        volume_t_4 = df['volume'].iloc[i-4] if i >= 4 else np.nan
        
        # Gap Efficiency & Range Momentum Framework
        # Multi-timeframe Gap Analysis
        short_term_gap_eff = (open_t - close_t_1) / (max(high_t_2, high_t_1, high_t) - min(low_t_2, low_t_1, low_t) + epsilon)
        medium_term_gap_eff = (open_t - close_t_1) / (max(high_t_4, high_t_3, high_t_2, high_t_1, high_t) - min(low_t_4, low_t_3, low_t_2, low_t_1, low_t) + epsilon)
        gap_eff_divergence = short_term_gap_eff - medium_term_gap_eff
        
        # Range Momentum Analysis
        fast_range_momentum = ((high_t - low_t) / (high_t_3 - low_t_3 + epsilon)) - 1
        medium_range_momentum = ((high_t - low_t) / (high_t_8 - low_t_8 + epsilon)) - 1
        range_momentum_divergence = fast_range_momentum - medium_range_momentum
        
        # Gap-Range Integration
        gap_range_alignment = np.sign(gap_eff_divergence) * np.sign(range_momentum_divergence)
        volatility_enhanced_gap = abs(open_t - close_t_1) / (max(high_t - low_t, abs(high_t - close_t_1), abs(low_t - close_t_1)) + epsilon)
        
        # Volume-Momentum Conformation System
        # Directional Momentum Strength
        bullish_pressure = sum(max(0, df['close'].iloc[i-j] - df['close'].iloc[i-j-1]) for j in range(5) if i-j-1 >= 0)
        bearish_pressure = sum(max(0, df['close'].iloc[i-j-1] - df['close'].iloc[i-j]) for j in range(5) if i-j-1 >= 0)
        directional_imbalance = (bullish_pressure - bearish_pressure) / (bullish_pressure + bearish_pressure + epsilon)
        
        # Volume Dynamics
        volume_efficiency = amount_t / (high_t - low_t + epsilon)
        vol_eff_avg = np.mean([df['amount'].iloc[j] / (df['high'].iloc[j] - df['low'].iloc[j] + epsilon) for j in range(i-2, i+1) if j >= 0])
        volume_flow_acceleration = volume_efficiency / (vol_eff_avg + epsilon)
        volume_price_alignment = np.sign(volume_flow_acceleration) * np.sign(close_t - close_t_3)
        
        # Momentum Fracture Detection
        price_acceleration = (close_t - close_t_1) / (close_t_1 - close_t_2 + epsilon)
        fracture_threshold = 1 if abs(price_acceleration) > 2.0 else 0
        
        # Intraday Pressure & Efficiency Metrics
        gap_pressure = (open_t - close_t_1) / (close_t_1 + epsilon)
        gap_momentum_alignment = np.sign(gap_pressure) * np.sign(close_t - open_t)
        daily_range_asymmetry = (close_t - low_t) / (high_t - low_t + epsilon) - 0.5
        range_efficiency = abs(close_t - close_t_1) / (high_t - low_t + epsilon)
        volume_concentration = volume_t / (high_t - low_t + epsilon)
        volume_surge = volume_t / (np.mean([df['volume'].iloc[j] for j in range(i-4, i+1) if j >= 0]) + epsilon)
        
        # Multi-Timeframe Fracture Patterns
        short_term_momentum = np.sign(close_t - close_t_1)
        medium_term_momentum = np.sign(close_t - close_t_5)
        multi_scale_fracture = 1 if (fracture_threshold and (short_term_momentum != medium_term_momentum)) else 0
        
        volume_momentum_alignment = np.sign(volume_flow_acceleration) * np.sign(range_momentum_divergence)
        
        # Adaptive Alpha Construction
        if abs(directional_imbalance) > 0.3:
            # High Momentum Regime
            gap_range_fracture_score = volatility_enhanced_gap * np.sign(open_t - close_t_1) * np.sign(range_momentum_divergence)
            efficiency_momentum_score = gap_eff_divergence * volume_momentum_alignment
            acceleration_differential = price_acceleration - (close_t_1 - close_t_2) / (close_t_2 - close_t_3 + epsilon)
            alpha = gap_range_fracture_score * efficiency_momentum_score * fracture_threshold * acceleration_differential
        else:
            # Low Momentum Regime
            volume_efficiency_integration = volume_price_alignment * volume_surge * range_efficiency * volume_concentration
            gap_range_confirmation = gap_range_alignment * gap_momentum_alignment
            alpha = volume_efficiency_integration * gap_range_confirmation * multi_scale_fracture
        
        result.iloc[i] = alpha
    
    return result
