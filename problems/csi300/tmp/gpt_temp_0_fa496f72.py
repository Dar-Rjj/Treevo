import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Quantum Cascade State Evolution
    cascade_wave_function = (df['volume'] + df['amount']) / (df['high'] - df['low']).replace(0, np.nan)
    cascade_uncertainty = np.abs(df['volume'] - df['amount']) / cascade_wave_function.replace(0, np.nan)
    quantum_cascade_collapse = np.abs(df['close'] - df['open']) / cascade_uncertainty.replace(0, np.nan)
    
    # Calculate rolling correlations for cascade entanglement
    cascade_entanglement = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window = df.iloc[i-20:i]
        corr = np.corrcoef(window['volume'], np.abs(window['close'] - window['open']))[0,1]
        cascade_entanglement.iloc[i] = corr if not np.isnan(corr) else 0
    
    # Price-Induced Cascade Interference
    close_5d_avg = df['close'].rolling(window=5, min_periods=1).mean()
    price_probability_amplitude = df['close'] / close_5d_avg.replace(0, np.nan)
    price_phase_shift = df['close'] / df['close'].shift(1).replace(0, np.nan)
    
    close_3d_median = df['close'].rolling(window=3, min_periods=1).median()
    quantum_price_collapse = np.abs(df['close'] - close_3d_median) / cascade_uncertainty.replace(0, np.nan)
    price_cascade_interference = price_probability_amplitude * cascade_uncertainty
    
    # Topological Entropic Cascade
    volume_5d_avg = df['volume'].rolling(window=5, min_periods=1).mean()
    information_flow_entropy = df['volume'] / volume_5d_avg.replace(0, np.nan)
    cascade_path_complexity = np.abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    close_10d_range = df['close'].rolling(window=10, min_periods=1).max() - df['close'].rolling(window=10, min_periods=1).min()
    price_distribution_entropy = df['close'] / close_10d_range.replace(0, np.nan)
    
    entropy_production_rate = information_flow_entropy * cascade_path_complexity
    information_dissipation = price_distribution_entropy / cascade_uncertainty.replace(0, np.nan)
    cascade_channel_capacity = information_flow_entropy * price_distribution_entropy
    entropic_efficiency = np.abs(df['close'] - df['open']) / (df['volume'] / df['amount']).replace(0, np.nan)
    
    # Multi-Scale Information Coherence
    price_coherence = np.abs(df['close'] - (df['high'] + df['low'])/2) / (df['high'] - df['low']).replace(0, np.nan)
    
    def harmonic_mean(series):
        return len(series) / np.sum(1/series.replace(0, np.nan))
    
    volume_3d_hmean = df['volume'].rolling(window=3, min_periods=1).apply(harmonic_mean, raw=False)
    volume_coherence = df['volume'] / volume_3d_hmean.replace(0, np.nan)
    quantum_decoherence = np.abs(price_coherence - volume_coherence)
    
    # Fractal Quantum Information Patterns
    high_low_range = df['high'] - df['low']
    range_5d_avg = high_low_range.rolling(window=5, min_periods=1).mean()
    fractal_complexity = high_low_range.rolling(window=5, min_periods=1).std() / range_5d_avg.replace(0, np.nan)
    
    bid_ask_pressure = (df['close'] - df['low']) / (df['high'] - df['close']).replace(0, np.nan)
    depth_fractal = (df['high'] - df['open']) / (df['open'] - df['low']).replace(0, np.nan)
    critical_asymmetry = np.abs(bid_ask_pressure - depth_fractal)
    
    # Causal Quantum Information Mechanics
    range_volume_causality = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        window = df.iloc[i-10:i]
        corr = np.corrcoef(window['high'] - window['low'], window['volume'].shift(1).fillna(0))[0,1]
        range_volume_causality.iloc[i] = corr if not np.isnan(corr) else 0
    
    # Quantum Information Transition Quality
    range_10d_avg = high_low_range.rolling(window=10, min_periods=1).mean()
    alignment_score = price_coherence * high_low_range / range_10d_avg.replace(0, np.nan)
    divergence_measure = np.abs(cascade_path_complexity - quantum_decoherence)
    
    # Regime classification
    quantum_information_regime = pd.Series('normal', index=df.index)
    quantum_information_regime[high_low_range > 1.5 * range_10d_avg] = 'high'
    quantum_information_regime[high_low_range < 0.7 * range_10d_avg] = 'low'
    
    # Regime-specific weights
    regime_weights = pd.Series(1.0, index=df.index)
    regime_weights[quantum_information_regime == 'high'] = 1.3
    regime_weights[quantum_information_regime == 'low'] = 0.7
    
    # Core Quantum Information Signal
    quantum_momentum = quantum_cascade_collapse * cascade_entanglement
    information_fractal = entropy_production_rate * fractal_complexity
    causal_quantum = range_volume_causality * alignment_score
    
    # Base signal
    base_signal = quantum_momentum * information_fractal * causal_quantum
    
    # Multi-scale validation
    scale_coherence = cascade_entanglement * fractal_complexity
    regime_consistency = regime_weights * alignment_score
    volume_confirmation = cascade_channel_capacity * entropic_efficiency
    
    # Final alpha factor
    validation_enhanced = base_signal * scale_coherence * regime_consistency
    alpha_factor = validation_enhanced * volume_confirmation * regime_weights
    
    # Clean infinite values and normalize
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_factor
