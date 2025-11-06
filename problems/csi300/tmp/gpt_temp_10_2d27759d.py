import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper functions
    def rolling_std(series, window):
        return series.rolling(window=window, min_periods=1).std()
    
    def rolling_mean(series, window):
        return series.rolling(window=window, min_periods=1).mean()
    
    # 1. Gap-Memory Dynamics Analysis
    # Short-Term Gap Memory (3-day)
    gap_return_3d = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_memory_acceleration = gap_return_3d - rolling_mean(gap_return_3d, 3)
    gap_memory_score = np.sign(gap_return_3d) * np.abs(gap_memory_acceleration)
    
    volume_memory_3d = data['volume'] / rolling_mean(data['volume'], 3) - 1
    gap_volume_divergence_score = np.sign(gap_return_3d) * (gap_return_3d - volume_memory_3d)
    
    gap_divergence_memory_change = gap_volume_divergence_score - rolling_mean(gap_volume_divergence_score, 3)
    
    # Calculate sign persistence
    def sign_persistence(series, window):
        signs = np.sign(series)
        persistence = pd.Series(0, index=series.index)
        for i in range(len(series)):
            if i >= window - 1:
                window_signs = signs.iloc[i-window+1:i+1]
                persistence.iloc[i] = (window_signs == window_signs.iloc[-1]).sum()
        return persistence
    
    gap_divergence_strength = np.abs(gap_divergence_memory_change) * sign_persistence(gap_volume_divergence_score, 3)
    
    # Medium-Term Efficiency Memory (10-day)
    daily_range_efficiency = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    efficiency_memory_momentum = daily_range_efficiency / rolling_mean(daily_range_efficiency, 10) - 1
    efficiency_memory_score = np.sign(efficiency_memory_momentum) * np.abs(efficiency_memory_momentum)
    
    gap_memory_return_10d = (data['open'] - rolling_mean(data['close'].shift(1), 10)) / rolling_mean(data['close'].shift(1), 10)
    gap_efficiency_divergence_score = np.sign(gap_memory_return_10d) * (gap_memory_return_10d - efficiency_memory_momentum)
    
    efficiency_divergence_change = gap_efficiency_divergence_score - rolling_mean(gap_efficiency_divergence_score, 5)
    efficiency_memory_strength = np.abs(efficiency_divergence_change) * sign_persistence(gap_efficiency_divergence_score, 5)
    
    # Cross-Timeframe Gap-Memory Analysis
    gap_memory_sign_alignment = np.sign(gap_volume_divergence_score) * np.sign(gap_efficiency_divergence_score)
    gap_memory_magnitude_ratio = np.abs(gap_volume_divergence_score) / (np.abs(gap_efficiency_divergence_score) + 0.001)
    cross_timeframe_consistency = np.sign(gap_divergence_memory_change) * np.sign(gap_memory_return_10d)
    
    # 2. Volatility-Regime Memory Detection
    # Multi-Scale Volatility Memory
    realized_vol_5d = rolling_std(data['close'], 5)
    
    gap_returns = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_vol_5d = rolling_std(gap_returns, 5)
    
    realized_vol_20d = rolling_std(data['close'], 20)
    gap_vol_20d = rolling_std(gap_returns, 20)
    
    volatility_compression_ratio = realized_vol_5d / (realized_vol_20d + 1e-8)
    gap_memory_stability_ratio = gap_vol_5d / (gap_vol_20d + 1e-8)
    
    # Memory-Based Regime Classification
    gap_memory_expansion_strength = volatility_compression_ratio * gap_memory_stability_ratio
    gap_memory_compression_strength = (1 / (volatility_compression_ratio + 1e-8)) * gap_memory_stability_ratio
    gap_memory_regime_stability = 1 - np.abs(volatility_compression_ratio - 1)
    
    # 3. Memory-Enhanced Factor Construction by Volatility Regime
    # High Volatility Memory Strategy
    gap_range_memory_ratio = np.abs(data['open'] - data['close'].shift(1)) / rolling_mean(np.abs(data['open'].shift(1) - data['close'].shift(2)), 5)
    
    def count_condition(series, window, threshold):
        count = pd.Series(0, index=series.index)
        for i in range(len(series)):
            if i >= window - 1:
                window_data = series.iloc[i-window+1:i+1]
                count.iloc[i] = (window_data > threshold).sum()
        return count
    
    gap_range_persistence = count_condition(gap_range_memory_ratio, 5, 1.2)
    
    volatility_scaled_gap_divergence = gap_volume_divergence_score * realized_vol_5d
    gap_memory_acceleration_breakout = gap_divergence_memory_change * gap_range_persistence
    
    gap_memory_acceleration_impact = gap_memory_acceleration * gap_memory_expansion_strength
    cross_volatility_gap_divergence = np.sign(gap_volume_divergence_score) * np.sign(gap_vol_5d - gap_vol_20d)
    
    high_vol_strategy = (volatility_scaled_gap_divergence + gap_memory_acceleration_breakout + 
                        gap_memory_acceleration_impact + cross_volatility_gap_divergence) / 4
    
    # Low Volatility Memory Strategy
    gap_volatility_gap = gap_vol_5d - gap_vol_20d
    efficiency_volatility_gap = rolling_std(daily_range_efficiency, 5) - rolling_std(daily_range_efficiency, 20)
    gap_memory_compression_alignment = np.sign(gap_volatility_gap) * np.sign(efficiency_volatility_gap)
    
    compression_strengthened_gap_memory = gap_volume_divergence_score * gap_memory_compression_strength
    gap_memory_persistence_low_vol = gap_divergence_strength * gap_memory_regime_stability
    
    range_constrained_gap_acceleration = gap_divergence_memory_change / (data['high'] - data['low'] + 0.001)
    efficiency_confirmation_compression = efficiency_memory_momentum * gap_memory_compression_alignment
    
    low_vol_strategy = (compression_strengthened_gap_memory + gap_memory_persistence_low_vol + 
                       range_constrained_gap_acceleration + efficiency_confirmation_compression) / 4
    
    # Transition Volatility Memory Strategy
    high_vol_weight = volatility_compression_ratio - 1
    low_vol_weight = 1 - volatility_compression_ratio
    
    # Initialize blended signal
    transition_strategy = pd.Series(0, index=data.index)
    
    # 4. Volume-Amount Memory Efficiency
    volume_concentration_memory = data['volume'] / (data['amount'] + 1e-8)
    absolute_gap_volume_impact = np.abs(data['open'] - data['close'].shift(1)) * data['volume']
    gap_liquidity_efficiency_ratio = absolute_gap_volume_impact / (rolling_mean(absolute_gap_volume_impact.shift(1), 5) + 0.001)
    
    # 5. Composite Gap-Memory Factor Synthesis
    factor = pd.Series(0, index=data.index)
    
    for i in range(len(data)):
        if i < 20:  # Ensure enough data for calculations
            continue
            
        vol_ratio = volatility_compression_ratio.iloc[i]
        
        if vol_ratio > 1.2:
            # High volatility regime
            regime_factor = high_vol_strategy.iloc[i] * gap_liquidity_efficiency_ratio.iloc[i]
        elif vol_ratio < 0.8:
            # Low volatility regime
            regime_factor = low_vol_strategy.iloc[i] * volume_concentration_memory.iloc[i]
        else:
            # Transition regime
            if i > 0:
                prev_factor = factor.iloc[i-1] if not pd.isna(factor.iloc[i-1]) else 0
                base_transition = (high_vol_weight.iloc[i] * high_vol_strategy.iloc[i] + 
                                 low_vol_weight.iloc[i] * low_vol_strategy.iloc[i]) / 2
                transition_factor = base_transition * (1 + gap_liquidity_efficiency_ratio.iloc[i])
                regime_factor = prev_factor * gap_memory_regime_stability.iloc[i] + transition_factor
            else:
                regime_factor = 0
        
        # Final adjustments
        gap_memory_acceleration_multiplier = 1 + 0.3 * np.sign(gap_memory_acceleration.iloc[i])
        efficiency_momentum_alignment = np.sign(efficiency_memory_momentum.iloc[i])
        
        factor.iloc[i] = regime_factor * gap_memory_acceleration_multiplier * efficiency_momentum_alignment
    
    # Normalize the factor
    factor = (factor - factor.mean()) / (factor.std() + 1e-8)
    
    return factor
