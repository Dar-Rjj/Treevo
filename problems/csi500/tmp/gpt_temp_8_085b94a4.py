import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required intermediate columns
    df['close_ret'] = df['close'] / df['close'].shift(1) - 1
    df['volume_ret'] = df['volume'] / df['volume'].shift(1) - 1
    
    for i in range(2, len(df)):
        # Fractal Momentum Structure
        # Price Fractal Momentum
        upper_fractal_momentum = (df['high'].iloc[i] - max(df['high'].iloc[i-1], df['high'].iloc[i-2])) * df['close_ret'].iloc[i]
        lower_fractal_momentum = (df['low'].iloc[i] - min(df['low'].iloc[i-1], df['low'].iloc[i-2])) * df['close_ret'].iloc[i]
        fractal_symmetry_momentum = (upper_fractal_momentum + lower_fractal_momentum) / 2
        
        # Volume Fractal Alignment
        gap_persistence_volume = ((df['open'].iloc[i] / df['close'].iloc[i-1]) / (df['open'].iloc[i-1] / df['close'].iloc[i-2])) * df['volume_ret'].iloc[i]
        fractal_volume_efficiency = ((df['high'].iloc[i] - df['low'].iloc[i]) / df['volume'].iloc[i]) * (df['volume'].iloc[i] / df['volume'].iloc[i-1])
        volume_fractal_score = gap_persistence_volume * fractal_volume_efficiency
        
        # Momentum-Fractal Integration
        price_volume_fractal = fractal_symmetry_momentum * volume_fractal_score
        if i > 2:
            prev_price_volume_fractal = (df['high'].iloc[i-1] - max(df['high'].iloc[i-2], df['high'].iloc[i-3])) * df['close_ret'].iloc[i-1]
            prev_price_volume_fractal += (df['low'].iloc[i-1] - min(df['low'].iloc[i-2], df['low'].iloc[i-3])) * df['close_ret'].iloc[i-1]
            prev_price_volume_fractal = prev_price_volume_fractal / 2 * volume_fractal_score
            fractal_decay_ratio = price_volume_fractal / prev_price_volume_fractal - 1 if prev_price_volume_fractal != 0 else 0
        else:
            fractal_decay_ratio = 0
        integrated_fractal_momentum = fractal_symmetry_momentum * fractal_decay_ratio
        
        # Volume-Price Efficiency Dynamics
        # Efficiency Gradient Analysis
        range_efficiency_current = (df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i]) if (df['high'].iloc[i] - df['low'].iloc[i]) != 0 else 0
        range_efficiency_prev = (df['close'].iloc[i-1] - df['open'].iloc[i-1]) / (df['high'].iloc[i-1] - df['low'].iloc[i-1]) if (df['high'].iloc[i-1] - df['low'].iloc[i-1]) != 0 else 0
        range_efficiency_momentum = range_efficiency_current / range_efficiency_prev if range_efficiency_prev != 0 else 0
        
        overnight_efficiency_current = abs(df['open'].iloc[i] / df['close'].iloc[i-1] - 1)
        overnight_efficiency_prev = abs(df['open'].iloc[i-1] / df['close'].iloc[i-2] - 1)
        overnight_efficiency_persistence = overnight_efficiency_current / overnight_efficiency_prev if overnight_efficiency_prev != 0 else 0
        
        efficiency_regime_score = range_efficiency_momentum * overnight_efficiency_persistence
        
        # Volume Distribution Dynamics
        volume_clustering_ratio = (df['volume'].iloc[i] / (df['volume'].iloc[i-1] + df['volume'].iloc[i-2])) * (df['volume'].iloc[i-1] / df['volume'].iloc[i-2]) if (df['volume'].iloc[i-1] + df['volume'].iloc[i-2]) != 0 and df['volume'].iloc[i-2] != 0 else 0
        
        amount_efficiency_current = df['amount'].iloc[i] / (df['volume'].iloc[i] * df['close'].iloc[i]) if (df['volume'].iloc[i] * df['close'].iloc[i]) != 0 else 0
        amount_efficiency_prev = df['amount'].iloc[i-1] / (df['volume'].iloc[i-1] * df['close'].iloc[i-1]) if (df['volume'].iloc[i-1] * df['close'].iloc[i-1]) != 0 else 0
        amount_efficiency_gradient = amount_efficiency_current / amount_efficiency_prev if amount_efficiency_prev != 0 else 0
        
        volume_distribution_momentum = volume_clustering_ratio * amount_efficiency_gradient
        
        # Efficiency-Volume Integration
        regime_efficiency_score = efficiency_regime_score * volume_distribution_momentum
        efficiency_volume_divergence = regime_efficiency_score / volume_fractal_score if volume_fractal_score != 0 else 0
        volume_efficiency_momentum = regime_efficiency_score * efficiency_volume_divergence
        
        # Momentum Decay Fractal Patterns
        # Price Momentum Decay Structure
        multi_period_momentum = df['close_ret'].iloc[i] / df['close_ret'].iloc[i-1] if df['close_ret'].iloc[i-1] != 0 else 0
        fractal_momentum_decay = fractal_symmetry_momentum / multi_period_momentum if multi_period_momentum != 0 else 0
        
        if i > 2:
            prev_fractal_symmetry = (df['high'].iloc[i-1] - max(df['high'].iloc[i-2], df['high'].iloc[i-3])) * df['close_ret'].iloc[i-1]
            prev_fractal_symmetry += (df['low'].iloc[i-1] - min(df['low'].iloc[i-2], df['low'].iloc[i-3])) * df['close_ret'].iloc[i-1]
            prev_fractal_symmetry /= 2
            prev_multi_period = df['close_ret'].iloc[i-1] / df['close_ret'].iloc[i-2] if df['close_ret'].iloc[i-2] != 0 else 0
            prev_fractal_decay = prev_fractal_symmetry / prev_multi_period if prev_multi_period != 0 else 0
            decay_persistence = fractal_momentum_decay / prev_fractal_decay if prev_fractal_decay != 0 else 0
        else:
            decay_persistence = 0
        
        # Volume Momentum Fractal Integration
        volume_momentum_decay = df['volume_ret'].iloc[i] / df['volume_ret'].iloc[i-1] if df['volume_ret'].iloc[i-1] != 0 else 0
        volume_fractal_decay = volume_fractal_score / volume_momentum_decay if volume_momentum_decay != 0 else 0
        volume_decay_alignment = volume_fractal_decay * volume_distribution_momentum
        
        # Decay-Fractal Convergence
        price_decay_fractal = decay_persistence * fractal_momentum_decay
        volume_decay_fractal = volume_decay_alignment * volume_fractal_decay
        combined_decay_fractal = price_decay_fractal * volume_decay_fractal
        
        # Volatility-Aware Signal Construction
        # Fractal Volatility Structure
        daily_fractal_range = (df['high'].iloc[i] - df['low'].iloc[i]) / (max(df['high'].iloc[i-1], df['high'].iloc[i-2]) - min(df['low'].iloc[i-1], df['low'].iloc[i-2])) if (max(df['high'].iloc[i-1], df['high'].iloc[i-2]) - min(df['low'].iloc[i-1], df['low'].iloc[i-2])) != 0 else 0
        
        overnight_fractal_current = abs(df['open'].iloc[i] - df['close'].iloc[i-1])
        overnight_fractal_prev = abs(df['open'].iloc[i-1] - df['close'].iloc[i-2])
        overnight_fractal_volatility = overnight_fractal_current / overnight_fractal_prev if overnight_fractal_prev != 0 else 0
        
        composite_fractal_volatility = daily_fractal_range * overnight_fractal_volatility
        
        # Signal Efficiency Integration
        raw_efficiency_signal = volume_efficiency_momentum * regime_efficiency_score
        volatility_scaled_efficiency = raw_efficiency_signal / composite_fractal_volatility if composite_fractal_volatility != 0 else 0
        efficiency_divergence_strength = volatility_scaled_efficiency * efficiency_volume_divergence
        
        # Momentum-Volatility Alignment
        momentum_volatility_signal = integrated_fractal_momentum / composite_fractal_volatility if composite_fractal_volatility != 0 else 0
        decay_volatility_alignment = combined_decay_fractal * composite_fractal_volatility
        volatility_adjusted_momentum = momentum_volatility_signal * decay_volatility_alignment
        
        # Final Alpha Construction
        # Core Divergence Factors
        primary_divergence = integrated_fractal_momentum * combined_decay_fractal
        secondary_divergence = volume_efficiency_momentum * efficiency_divergence_strength
        tertiary_divergence = volatility_adjusted_momentum * volume_decay_alignment
        
        # Risk-Aware Refinement
        volatility_weighted_core = primary_divergence / composite_fractal_volatility if composite_fractal_volatility != 0 else 0
        efficiency_weighted_signal = secondary_divergence * regime_efficiency_score
        momentum_filtered_output = tertiary_divergence * fractal_symmetry_momentum
        
        # Final Alpha Output
        core_alpha = volatility_weighted_core * efficiency_weighted_signal
        enhancement_factor = momentum_filtered_output * volume_distribution_momentum
        final_factor = core_alpha * enhancement_factor
        
        result.iloc[i] = final_factor
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
