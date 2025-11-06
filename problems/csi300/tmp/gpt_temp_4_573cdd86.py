import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate entropy measures
    def calculate_entropy(series, window=5):
        returns = series.pct_change().dropna()
        if len(returns) < window:
            return pd.Series(index=series.index, dtype=float)
        
        entropy_values = []
        for i in range(len(series)):
            if i < window:
                entropy_values.append(np.nan)
                continue
                
            window_returns = returns.iloc[i-window+1:i+1]
            if window_returns.std() == 0:
                entropy_values.append(0)
            else:
                # Simple entropy approximation using normalized variance
                normalized_var = window_returns.var() / (window_returns.abs().mean() + 1e-8)
                entropy_values.append(normalized_var)
        
        return pd.Series(entropy_values, index=series.index)
    
    # Calculate price and volume entropy
    price_entropy = calculate_entropy(df['close'])
    volume_entropy = calculate_entropy(df['volume'])
    
    # Multi-timeframe Entropy Compression
    price_entropy_compression = price_entropy / price_entropy.rolling(window=5, min_periods=1).mean()
    volume_entropy_compression = volume_entropy / volume_entropy.rolling(window=5, min_periods=1).mean()
    cross_entropy = (price_entropy - volume_entropy).abs()
    cross_entropy_compression = cross_entropy / cross_entropy.rolling(window=5, min_periods=1).mean()
    
    # Entropy Regime Classification
    high_compression_regime = ((price_entropy_compression < 0.8) & (volume_entropy_compression < 0.8)).astype(float)
    expansion_regime = ((price_entropy_compression > 1.2) | (volume_entropy_compression > 1.2)).astype(float)
    transition_regime = (cross_entropy_compression > 1.1).astype(float)
    
    # Entropy State Momentum
    entropy_compression_momentum = price_entropy_compression / price_entropy_compression.shift(1)
    cross_entropy_momentum = cross_entropy_compression / cross_entropy_compression.shift(1)
    
    # Regime persistence
    regime_persistence = pd.Series(0, index=df.index)
    current_regime = 0
    persistence_count = 0
    for i in range(len(df)):
        if high_compression_regime.iloc[i] == 1:
            new_regime = 1
        elif expansion_regime.iloc[i] == 1:
            new_regime = 2
        elif transition_regime.iloc[i] == 1:
            new_regime = 3
        else:
            new_regime = 0
            
        if new_regime == current_regime:
            persistence_count += 1
        else:
            persistence_count = 1
            current_regime = new_regime
            
        regime_persistence.iloc[i] = persistence_count
    
    # Gap-Entropy Elasticity Dynamics
    gap_magnitude = (df['open'] - df['close'].shift(1)).abs() / (df['close'].shift(1) + 1e-8)
    gap_magnitude_entropy = gap_magnitude * price_entropy
    
    volume_weighted_gap = (df['open'] - df['close'].shift(1)).abs() * df['volume'] / (df['amount'] + 1e-8)
    
    daily_range = df['high'] - df['low']
    entropy_gap_efficiency = daily_range / ((df['open'] - df['close'].shift(1)).abs() + 1e-8) * volume_entropy
    
    # Elasticity Regime Detection
    high_elasticity = ((entropy_gap_efficiency > 2.0) & (gap_magnitude_entropy > 0.01)).astype(float)
    low_elasticity = ((entropy_gap_efficiency < 0.5) | (gap_magnitude_entropy < 0.005)).astype(float)
    transition_elasticity = (volume_weighted_gap > volume_weighted_gap.rolling(window=5, min_periods=1).mean()).astype(float)
    
    # Gap-Entropy Coupling Patterns
    price_entropy_gap_coupling = (df['close'] - df['open']) * price_entropy / ((df['open'] - df['close'].shift(1)).abs() + 1e-8)
    volume_entropy_gap_coupling = df['volume'] * volume_entropy / ((df['open'] - df['close'].shift(1)).abs() * df['amount'] + 1e-8)
    cross_gap_entropy = (price_entropy_gap_coupling - volume_entropy_gap_coupling).abs()
    
    # Microstructure Efficiency-Elasticity
    price_efficiency_elasticity = ((df['close'] - df['open']) / (daily_range + 1e-8)) * price_entropy
    volume_efficiency_elasticity = (df['volume'] / (df['amount'] / (daily_range + 1e-8))) * volume_entropy
    cross_efficiency_elasticity = (price_efficiency_elasticity - volume_efficiency_elasticity).abs()
    
    # Efficiency-Compression Interaction
    compressed_efficiency = price_efficiency_elasticity * price_entropy_compression
    expanded_efficiency = volume_efficiency_elasticity * volume_entropy_compression
    efficiency_regime_alignment = compressed_efficiency * expanded_efficiency
    
    # Elasticity-Efficiency Coupling
    gap_efficiency_coupling = entropy_gap_efficiency * price_efficiency_elasticity
    volume_efficiency_coupling = volume_efficiency_elasticity * volume_entropy_gap_coupling
    cross_coupling_strength = (gap_efficiency_coupling - volume_efficiency_coupling).abs()
    
    # Momentum-Elasticity Synthesis
    price_momentum = df['close'] / df['close'].shift(1) - 1
    volume_momentum = df['volume'] / df['volume'].shift(1) - 1
    
    price_momentum_elasticity = price_momentum * price_entropy_gap_coupling
    volume_momentum_elasticity = volume_momentum * volume_entropy_gap_coupling
    cross_momentum_elasticity = (price_momentum_elasticity - volume_momentum_elasticity).abs()
    
    # Regime-Adaptive Momentum
    compression_momentum = price_momentum_elasticity * high_compression_regime
    expansion_momentum = volume_momentum_elasticity * expansion_regime
    transition_momentum = cross_momentum_elasticity * transition_regime
    
    # Efficiency-Momentum Alignment
    efficient_momentum = price_momentum_elasticity * price_efficiency_elasticity
    inefficient_momentum = volume_momentum_elasticity * volume_efficiency_elasticity
    momentum_regime_persistence = efficient_momentum / (inefficient_momentum + 1e-8)
    
    # Composite Alpha Construction
    # Primary Component: Entropy-compression regime momentum
    primary_component = entropy_compression_momentum * regime_persistence
    
    # Secondary Component: Gap-entropy elasticity dynamics
    secondary_component = entropy_gap_efficiency * cross_gap_entropy
    
    # Tertiary Component: Efficiency-elasticity coupling
    tertiary_component = efficiency_regime_alignment * cross_coupling_strength
    
    # Quaternary Component: Regime-adaptive momentum alignment
    quaternary_component = (compression_momentum + expansion_momentum + transition_momentum) * momentum_regime_persistence
    
    # Final Factor Synthesis
    # Combine entropy-compression with gap-elasticity
    factor_stage1 = primary_component * secondary_component
    
    # Apply efficiency-elasticity filters
    efficiency_filter = np.where(efficiency_regime_alignment > 0, 1, -1)
    factor_stage2 = factor_stage1 * efficiency_filter
    
    # Incorporate regime-adaptive momentum signals
    momentum_signal = np.sign(quaternary_component)
    factor_stage3 = factor_stage2 * momentum_signal
    
    # Add tertiary component for final refinement
    final_factor = factor_stage3 + 0.3 * tertiary_component
    
    return final_factor
