import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Quantum Asymmetric Memory Dynamics with Fractal Entropy Integration
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper functions for fractal dimensions and entropy
    def compute_hurst_exponent(series, max_lag=20):
        """Compute Hurst exponent as proxy for fractal dimension"""
        if len(series) < max_lag:
            return 0.5
        lags = range(2, min(max_lag, len(series)//2))
        tau = []
        for lag in lags:
            series_lag = series.diff(lag).dropna()
            if len(series_lag) > 0:
                tau.append(np.std(series_lag))
            else:
                tau.append(0)
        if len(tau) < 2:
            return 0.5
        try:
            hurst = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)[0]
            return max(0.1, min(0.9, hurst))
        except:
            return 0.5
    
    def compute_entropy(series, window=10):
        """Compute Shannon entropy of price changes"""
        if len(series) < window:
            return 0.2
        returns = series.pct_change().dropna()
        if len(returns) < window:
            return 0.2
        recent_returns = returns.tail(window)
        if len(recent_returns) == 0:
            return 0.2
        hist, _ = np.histogram(recent_returns, bins=5, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.2
        entropy = -np.sum(hist * np.log(hist))
        return max(0.1, min(1.0, entropy / np.log(len(hist))))
    
    # Calculate fractal dimensions and entropy measures
    data['price_fractal'] = data['close'].rolling(window=20, min_periods=10).apply(
        lambda x: compute_hurst_exponent(x), raw=False
    ).fillna(0.5)
    
    data['volume_fractal'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: compute_hurst_exponent(x), raw=False
    ).fillna(0.5)
    
    data['microstructure_fractal'] = (data['price_fractal'] + data['volume_fractal']) / 2
    
    data['price_entropy'] = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: compute_entropy(x), raw=False
    ).fillna(0.2)
    
    data['volume_entropy'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: compute_entropy(x), raw=False
    ).fillna(0.2)
    
    # Asymmetric Fractal Framework
    data['fractal_upward_vol'] = ((data['high'] - np.maximum(data['open'], data['close'])) / 
                                 data['close'].shift(1)) * data['price_fractal']
    data['fractal_downward_vol'] = ((np.minimum(data['open'], data['close']) - data['low']) / 
                                   data['close'].shift(1)) * data['price_fractal']
    data['entropic_vol_asymmetry'] = (data['fractal_upward_vol'] - data['fractal_downward_vol']) * data['price_entropy']
    data['volume_fractal_asymmetry'] = data['entropic_vol_asymmetry'] * data['volume_fractal']
    
    # Memory-Enhanced Fractal Patterns
    data['fractal_high_memory'] = (data['high'].rolling(window=3, min_periods=3).max() * 
                                  data['microstructure_fractal'])
    data['fractal_low_memory'] = (data['low'].rolling(window=3, min_periods=3).min() * 
                                 data['microstructure_fractal'])
    
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, 1e-6)  # Avoid division by zero
    
    data['entropic_memory_break'] = ((data['high'] - data['fractal_high_memory']) / 
                                    high_low_range) * data['price_entropy']
    data['fractal_memory_hold'] = ((data['fractal_low_memory'] - data['low']) / 
                                  high_low_range) * data['volume_entropy']
    
    # Quantum Asymmetric States
    data['high_entropy_asymmetry'] = ((data['entropic_vol_asymmetry'] > 0.1) & 
                                     (data['price_entropy'] > 0.2)).astype(float)
    data['low_entropy_memory'] = ((data['fractal_memory_hold'] > 0.05) & 
                                 (data['volume_entropy'] < 0.1)).astype(float)
    data['coherent_asymmetric_break'] = ((data['entropic_memory_break'] > 0.1) & 
                                        (data['microstructure_fractal'] > 0.8)).astype(float)
    
    # Entropic Memory Momentum Framework
    data['memory_break_momentum'] = (data['entropic_memory_break'] * 
                                    (data['close'] / data['close'].shift(1) - 1))
    data['volume_memory_persistence'] = (data['fractal_memory_hold'] * 
                                        (data['volume'] / data['volume'].shift(1) - 1))
    data['asymmetric_fractal_momentum'] = (data['volume_fractal_asymmetry'] * 
                                          (data['microstructure_fractal'] / data['microstructure_fractal'].shift(1) - 1))
    
    # Quantum Memory Microstructure
    data['memory_price_entanglement'] = (np.abs(data['close'] - data['open']) * 
                                        data['volume'] * data['entropic_memory_break'])
    data['quantum_memory_stress'] = data['memory_price_entanglement'] / high_low_range
    
    # Quantum Memory Reversal Framework
    data['quantum_memory_upside_reversal'] = (data['high'] - np.maximum(data['open'], data['close'])) * data['entropic_memory_break']
    data['quantum_memory_downside_reversal'] = (np.minimum(data['open'], data['close']) - data['low']) * data['fractal_memory_hold']
    data['net_quantum_memory_reversal'] = data['quantum_memory_upside_reversal'] - data['quantum_memory_downside_reversal']
    
    # Core Quantum Memory Components
    data['entropic_memory_velocity'] = (data['memory_break_momentum'] * 
                                       data['microstructure_fractal'] * data['entropic_memory_break'])
    
    # Quantum Memory Reversal Persistence
    def compute_persistence(series, window=3):
        if len(series) < window + 1:
            return 0.5
        signs = np.sign(series)
        persistence = []
        for i in range(len(signs) - window, len(signs) - 1):
            if i >= 1:
                persistence.append(float(signs.iloc[i] == signs.iloc[i-1]))
        return np.mean(persistence) if persistence else 0.5
    
    data['quantum_memory_reversal_persistence'] = data['net_quantum_memory_reversal'].rolling(
        window=4, min_periods=3
    ).apply(compute_persistence, raw=False).fillna(0.5)
    
    data['quantum_memory_reversal_momentum'] = (data['net_quantum_memory_reversal'] * 
                                               (data['quantum_memory_upside_reversal'] / 
                                                (data['quantum_memory_downside_reversal'] + 1e-6)) * 
                                               data['quantum_memory_reversal_persistence'])
    
    data['volume_memory_dynamics'] = (data['volume_memory_persistence'] * 
                                     (data['volume'] / data['volume'].shift(1)) * data['fractal_memory_hold'])
    
    data['quantum_memory_breakout_velocity'] = ((data['fractal_high_memory'] / data['high'].shift(1) - 
                                                data['fractal_low_memory'] / data['low'].shift(1)) * 
                                               data['quantum_memory_stress'] * 
                                               (high_low_range / (data['high'].shift(1) - data['low'].shift(1) + 1e-6)))
    
    # Quantum Memory State Enhancement
    enhancement_factors = pd.Series(1.0, index=data.index)
    enhancement_factors[data['high_entropy_asymmetry'] == 1] *= 1.4
    enhancement_factors[data['low_entropy_memory'] == 1] *= 1.3
    enhancement_factors[data['coherent_asymmetric_break'] == 1] *= 1.2
    
    # Apply stress modulation
    stress_modulation = np.clip(data['quantum_memory_stress'] * data['microstructure_fractal'], 0.5, 2.0)
    
    # Validated Quantum Memory Integration
    data['coherence_confirmed_memory'] = (data['entropic_memory_velocity'] * 
                                         np.sign(data['memory_break_momentum']) * np.sign(data['asymmetric_fractal_momentum']))
    
    data['volume_enhanced_quantum_memory'] = (data['quantum_memory_reversal_momentum'] * 
                                             data['volume_memory_persistence'].rolling(window=4, min_periods=3).apply(
                                                 compute_persistence, raw=False).fillna(0.5))
    
    data['fractal_memory_breakout_momentum'] = (data['volume_memory_dynamics'] * 
                                               data['asymmetric_fractal_momentum'].rolling(window=4, min_periods=3).apply(
                                                   compute_persistence, raw=False).fillna(0.5))
    
    # Memory coherence signals
    high_coherence = ((data['microstructure_fractal'] > 0.9) & 
                     (data['entropic_memory_break'] < 0.8)).astype(float)
    low_coherence = ((data['microstructure_fractal'] < 0.5) & 
                    (data['fractal_memory_hold'] > 1.2)).astype(float)
    
    data['quantum_memory_range_dynamics'] = (data['quantum_memory_breakout_velocity'] * 
                                            (high_coherence - low_coherence))
    
    # Persistence measures
    data['entropic_memory_persistence'] = data['entropic_memory_break'].rolling(
        window=4, min_periods=3).apply(compute_persistence, raw=False).fillna(0.5)
    
    data['quantum_memory_stress_persistence'] = (data['quantum_memory_stress'] * data['microstructure_fractal']).rolling(
        window=4, min_periods=3).apply(compute_persistence, raw=False).fillna(0.5)
    
    # Final Quantum Memory Alpha components
    primary_factor = data['coherence_confirmed_memory'] * data['entropic_memory_persistence']
    secondary_factor = data['volume_enhanced_quantum_memory'] * data['quantum_memory_stress_persistence']
    tertiary_factor = data['fractal_memory_breakout_momentum'] * (np.sign(data['memory_break_momentum']) * np.sign(data['asymmetric_fractal_momentum']))
    
    # Entropic Memory Mean Reversion
    data['entropic_memory_mean_reversion'] = (1 - np.abs(data['close'] - data['close'].shift(1)) / high_low_range) * data['entropic_memory_break']
    quaternary_factor = data['quantum_memory_range_dynamics'] * data['entropic_memory_mean_reversion']
    
    # Composite Quantum Memory Alpha
    composite_alpha = (primary_factor * 0.35 + 
                      secondary_factor * 0.25 + 
                      tertiary_factor * 0.25 + 
                      quaternary_factor * 0.15)
    
    # Apply quantum state enhancements and stress modulation
    final_alpha = composite_alpha * enhancement_factors * stress_modulation
    
    # Normalize and clean
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Remove any potential lookahead bias by ensuring no forward-looking operations
    return final_alpha
