import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    
    for i in range(len(df)):
        if i < 8:  # Need at least 8 days of data
            alpha.iloc[i] = 0
            continue
            
        # Current values
        open_t = df['open'].iloc[i]
        high_t = df['high'].iloc[i]
        low_t = df['low'].iloc[i]
        close_t = df['close'].iloc[i]
        amount_t = df['amount'].iloc[i]
        volume_t = df['volume'].iloc[i]
        
        # Multi-scale Momentum Structure
        # Price Momentum Components
        close_t_3 = df['close'].iloc[i-3]
        close_t_8 = df['close'].iloc[i-8]
        
        short_price_momentum = close_t / close_t_3 - 1
        medium_price_momentum = close_t / close_t_8 - 1
        price_momentum_convergence = short_price_momentum - medium_price_momentum
        
        # Volume Momentum Components
        volume_t_3 = df['volume'].iloc[i-3]
        volume_t_8 = df['volume'].iloc[i-8]
        
        short_volume_momentum = volume_t / volume_t_3 - 1
        medium_volume_momentum = volume_t / volume_t_8 - 1
        volume_momentum_convergence = short_volume_momentum - medium_volume_momentum
        
        # Fractal Efficiency Analysis
        close_t_2 = df['close'].iloc[i-2]
        
        # Short-term fractal
        high_window_short = max(df['high'].iloc[i-2:i+1])
        low_window_short = min(df['low'].iloc[i-2:i+1])
        short_fractal = abs(close_t - close_t_2) / (high_window_short - low_window_short + eps)
        
        # Medium-term fractal
        high_window_medium = max(df['high'].iloc[i-8:i+1])
        low_window_medium = min(df['low'].iloc[i-8:i+1])
        medium_fractal = abs(close_t - close_t_8) / (high_window_medium - low_window_medium + eps)
        
        fractal_efficiency_divergence = short_fractal - medium_fractal
        
        # Volume-Price Dynamics
        volume_t_1 = df['volume'].iloc[i-1]
        volume_acceleration = (volume_t - volume_t_1) / (volume_t_1 + eps)
        price_momentum = (close_t - close_t_3) / (close_t_3 + eps)
        close_t_1 = df['close'].iloc[i-1]
        dynamic_alignment = volume_acceleration * price_momentum * abs(close_t - close_t_1)
        
        # Microstructure Quality Assessment
        price_impact_ratio = abs(close_t - open_t) / (high_t - low_t + eps)
        volume_value_consistency = amount_t / (volume_t * close_t + eps)
        trade_intensity = amount_t / (high_t - low_t + eps)
        quality_score = price_impact_ratio * volume_value_consistency
        
        # Gap & Pressure Analysis
        intraday_pressure = (close_t - (high_t + low_t) / 2) / (high_t - low_t + eps)
        
        high_window_gap = max(df['high'].iloc[i-1:i+1])
        low_window_gap = min(df['low'].iloc[i-1:i+1])
        close_t_1_gap = df['close'].iloc[i-1]
        overnight_gap_efficiency = (open_t - close_t_1_gap) / (high_window_gap - low_window_gap + eps)
        
        pressure_consistency = intraday_pressure * overnight_gap_efficiency
        
        # Convergence & Divergence Integration
        price_volume_divergence = price_momentum_convergence - volume_momentum_convergence
        fractal_momentum_alignment = price_momentum_convergence * fractal_efficiency_divergence
        dynamic_convergence_signal = price_volume_divergence * fractal_momentum_alignment
        
        # Adaptive Signal Processing
        core_signal = dynamic_convergence_signal * dynamic_alignment
        quality_filtered_signal = core_signal * quality_score
        microstructure_enhanced_signal = quality_filtered_signal * (1 + trade_intensity)
        pressure_adjusted_signal = microstructure_enhanced_signal * pressure_consistency
        
        # Final Alpha Synthesis
        if i >= 5:
            high_t_5 = df['high'].iloc[i-5]
            low_t_5 = df['low'].iloc[i-5]
            volatility_awareness = (high_t - low_t) / (high_t_5 - low_t_5 + eps)
        else:
            volatility_awareness = 1.0
            
        adaptive_scaling = pressure_adjusted_signal / (volatility_awareness + eps)
        
        # Final alpha with tanh normalization and momentum sign
        final_alpha = np.tanh(adaptive_scaling * np.sign(price_momentum_convergence))
        
        alpha.iloc[i] = final_alpha
    
    return alpha
