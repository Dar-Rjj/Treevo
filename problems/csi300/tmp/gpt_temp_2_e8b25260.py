import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Helper functions
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.abs(high - close_prev), np.abs(low - close_prev))
    
    # Fractal Range Entropy States
    # Range entropy
    close_prev = data['close'].shift(1)
    tr = true_range(data['high'], data['low'], close_prev)
    range_entropy = np.log(data['high'] - data['low'] + 1e-8) / np.log(tr + 1e-8)
    
    # Efficiency divergence
    efficiency_divergence = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    
    # Fractal range collapse
    range_change = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)
    range_collapse = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if range_change.iloc[i] < 0.5:
            range_collapse.iloc[i] = 0
        else:
            range_collapse.iloc[i] = range_collapse.iloc[i-1] + 1
    
    # Range multifractality
    morning_range = (data['high'] - data['open']) / data['open']
    afternoon_range = (data['close'] - data['low']) / data['low']
    range_multifractality = morning_range.rolling(window=5).var() / (afternoon_range.rolling(window=5).var() + 1e-8)
    
    # Analyze Fractal Liquidity Patterns
    # Volume entropy
    volume_entropy = np.log(data['volume'] + 1e-8) - np.log(data['volume'].shift(1) + 1e-8)
    
    # Large flow divergence (simplified)
    amount_rolling = data['amount'].rolling(window=20)
    large_flow_divergence = data['amount'] / (amount_rolling.quantile(0.95) + 1e-8)
    
    # Liquidity complexity
    liquidity_complexity = volume_entropy * range_entropy
    
    # Fractal liquidity collapse
    volume_change = data['volume'] / (data['volume'].shift(1) + 1e-8)
    liquidity_collapse = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if volume_change.iloc[i] > 2.0:  # Volume clustering threshold
            liquidity_collapse.iloc[i] = 0
        else:
            liquidity_collapse.iloc[i] = liquidity_collapse.iloc[i-1] + 1
    
    # Generate Fractal Range-Liquidity Signals
    fractal_stability = range_entropy * liquidity_complexity
    fractal_state_transition = range_collapse * large_flow_divergence
    fractal_measurement = efficiency_divergence * large_flow_divergence
    
    # Entropic Momentum Persistence
    # Momentum entropy
    close_5 = data['close'].shift(5)
    high_5d = data['high'].rolling(window=5).max()
    low_5d = data['low'].rolling(window=5).min()
    momentum_entropy = (data['close'] - close_5) / (high_5d - low_5d + 1e-8)
    
    # Fractal momentum cycles
    momentum_sign = np.sign(data['close'] - data['close'].shift(1))
    momentum_cycles = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if momentum_sign.iloc[i] != momentum_sign.iloc[i-1]:
            momentum_cycles.iloc[i] = momentum_cycles.iloc[i-1] + 1
        else:
            momentum_cycles.iloc[i] = 0
    
    # Entropic momentum density
    momentum_reversals = (momentum_sign != momentum_sign.shift(1)).rolling(window=10).sum()
    momentum_range = high_5d - low_5d
    entropic_momentum_density = momentum_reversals / (momentum_range + 1e-8)
    
    # Momentum fractal dimension
    momentum_regime = (data['close'] > data['close'].rolling(window=5).mean()).astype(int)
    momentum_fractal_dimension = (momentum_regime != momentum_regime.shift(1)).rolling(window=10).sum()
    
    # Analyze Volume-Induced Entropy
    # Volume entropy singularities
    volume_5d_mean = data['volume'].rolling(window=5).mean()
    volume_singularity = data['volume'] / (volume_5d_mean + 1e-8)
    
    # Momentum entropy persistence
    volume_persistence = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if volume_singularity.iloc[i] > 1.5:
            volume_persistence.iloc[i] = 0
        else:
            volume_persistence.iloc[i] = volume_persistence.iloc[i-1] + 1
    
    # Entropic momentum transitions
    entropic_momentum_transitions = range_collapse * volume_singularity
    
    # Momentum critical divergence
    high_low_divergence = (data['high'] - data['low']) / data['close']
    momentum_critical_divergence = high_low_divergence * volume_singularity
    
    # Generate Entropic Momentum Signals
    entropic_stability = momentum_entropy * volume_singularity
    entropy_deformation = momentum_fractal_dimension * range_collapse
    phase_divergence = momentum_critical_divergence * volume_singularity
    
    # Volatility-Entropy Fractal Transitions
    # Volatility divergence
    tr_7d_median = tr.rolling(window=7).median()
    volatility_divergence = tr / (tr_7d_median + 1e-8)
    
    # Volatility entropy
    volatility_entropy = tr.rolling(window=7).var()
    
    # Fractal volatility entanglement
    volatility_entanglement = volatility_divergence * efficiency_divergence
    
    # Volatility multifractality
    volatility_multifractality = tr.rolling(window=3).std() / (tr.rolling(window=10).std() + 1e-8)
    
    # Analyze Entropy Dynamics
    # Short and medium entropy
    short_entropy = (data['close'] - data['close'].shift(2)) / (data['close'].shift(2) + 1e-8)
    medium_entropy = (data['close'] - data['close'].shift(8)) / (data['close'].shift(8) + 1e-8)
    
    # Entropy convergence
    entropy_convergence = short_entropy * medium_entropy
    
    # Fractal entropy collapse
    entropy_scale = np.abs(short_entropy) / (np.abs(medium_entropy) + 1e-8)
    entropy_collapse = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if entropy_scale.iloc[i] > 2.0 or entropy_scale.iloc[i] < 0.5:
            entropy_collapse.iloc[i] = 0
        else:
            entropy_collapse.iloc[i] = entropy_collapse.iloc[i-1] + 1
    
    # Generate Volatility-Entropy Signals
    volatility_weighted_entropy = entropy_convergence * volatility_divergence
    fractal_scale_shift = volatility_entropy * entropy_convergence
    volatility_entropy_entanglement = fractal_stability * entropy_collapse
    
    # Intraday Fractal Efficiency
    # Morning and afternoon patterns
    morning_divergence = (data['high'] - data['open']) / (data['open'] + 1e-8)
    afternoon_convergence = (data['close'] - data['low']) / (data['low'] + 1e-8)
    
    morning_efficiency_divergence = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    afternoon_efficiency_convergence = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Analyze Intraday Entropy
    intraday_entropy = morning_divergence * (data['volume'] / data['volume'].rolling(window=5).mean())
    efficiency_flow_synergy = afternoon_efficiency_convergence * large_flow_divergence
    
    # Intraday fractal persistence
    intraday_persistence = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if morning_divergence.iloc[i] > afternoon_convergence.iloc[i]:
            intraday_persistence.iloc[i] = intraday_persistence.iloc[i-1] + 1
        else:
            intraday_persistence.iloc[i] = 0
    
    # Fractal intraday collapse
    intraday_pattern = morning_divergence - afternoon_convergence
    intraday_collapse = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if np.abs(intraday_pattern.iloc[i]) < 0.01:
            intraday_collapse.iloc[i] = 0
        else:
            intraday_collapse.iloc[i] = intraday_collapse.iloc[i-1] + 1
    
    # Generate Intraday Signals
    morning_fractal_divergence = morning_divergence * volume_singularity
    afternoon_entropic_convergence = afternoon_convergence * intraday_persistence
    intraday_entropic_alignment = intraday_persistence * entropy_convergence
    
    # Fractal-Entropic Integration
    # Fractal Range Efficiency Component
    fractal_range_stability = range_entropy * efficiency_divergence
    liquidity_fractal_states = volume_entropy * large_flow_divergence
    fractal_range_liquidity = fractal_range_stability * liquidity_fractal_states
    
    # Entropic Momentum Component
    adaptive_entropic_momentum = momentum_entropy * volume_singularity
    volume_validated_entropy = adaptive_entropic_momentum * volume_persistence
    entropic_momentum_strength = volume_validated_entropy * momentum_critical_divergence
    
    # Volatility-Entropy Fractal Component
    fractal_volatility_entropy = volatility_weighted_entropy * fractal_scale_shift
    volatility_entropy_entanglement_component = fractal_volatility_entropy * entropy_convergence
    fractal_transition_strength = volatility_entropy_entanglement_component * entropy_collapse
    
    # Intraday Fractal-Entropy
    intraday_fractal_efficiency = morning_fractal_divergence * afternoon_entropic_convergence
    fractal_intraday_persistence = intraday_fractal_efficiency * intraday_persistence
    intraday_entropic_alignment_component = fractal_intraday_persistence * entropy_convergence
    
    # Multi-Scale Fractal Validation
    # Fractal Scale Coherence
    range_3d = (data['high'] - data['low']).rolling(window=3).mean()
    range_10d = (data['high'] - data['low']).rolling(window=10).mean()
    multi_scale_range_entropy = range_3d / (range_10d + 1e-8)
    
    momentum_3d = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    momentum_10d = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    fractal_momentum_persistence = momentum_3d / (momentum_10d + 1e-8)
    
    volatility_3d = tr.rolling(window=3).std()
    volatility_10d = tr.rolling(window=10).std()
    volatility_fractal_consistency = volatility_3d / (volatility_10d + 1e-8)
    
    fractal_scale_coherence = multi_scale_range_entropy * fractal_momentum_persistence * volatility_fractal_consistency
    
    # Entropic Fractal Stability
    fractal_entropic_density = entropic_momentum_density * range_entropy
    momentum_fractal_entropy = momentum_entropy * fractal_momentum_persistence
    range_fractal_entropy = range_entropy * multi_scale_range_entropy
    
    entropic_stability_component = fractal_entropic_density * momentum_fractal_entropy * range_fractal_entropy
    
    # Fractal-Entropic Convergence
    scale_fractal_alignment = fractal_scale_coherence * entropic_stability_component
    momentum_fractal_entropy_component = momentum_fractal_entropy * range_fractal_entropy
    fractal_entropic_convergence = scale_fractal_alignment * momentum_fractal_entropy_component
    
    # Final Fractal Entropic Alpha
    # Core Fractal-Entropic Signal
    fractal_range_momentum = fractal_range_liquidity * entropic_momentum_strength
    volatility_fractal_entropy_component = fractal_transition_strength * intraday_entropic_alignment_component
    fractal_entropic_integration = fractal_range_momentum * volatility_fractal_entropy_component
    
    # Multi-Scale Fractal Validation
    fractal_scale_validation = fractal_entropic_integration * fractal_entropic_convergence
    entropic_fractal_coherence = fractal_scale_validation * fractal_momentum_persistence
    fractal_entropic_stability = entropic_fractal_coherence * efficiency_divergence
    
    # Fractal Entropic Momentum Dynamics Alpha
    base_fractal_factor = fractal_entropic_integration * fractal_scale_validation
    fractal_entropic_enhancement = base_fractal_factor * fractal_entropic_stability
    final_alpha = fractal_entropic_enhancement * intraday_entropic_alignment_component
    
    return final_alpha.fillna(0)
