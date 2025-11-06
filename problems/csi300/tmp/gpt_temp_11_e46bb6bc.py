import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Helper functions for fractal and entropy calculations
    def price_fractal_dimension(high, low, close, window=5):
        """Calculate price fractal dimension using Hurst exponent approximation"""
        returns = np.log(close).diff()
        L = np.abs(returns).rolling(window=window).mean()
        S = returns.rolling(window=window).std()
        H = np.log(L / S) / np.log(window)
        return 2 - H
    
    def price_entropy(close, window=10):
        """Calculate price entropy using Shannon entropy on returns"""
        returns = np.log(close).diff().dropna()
        if len(returns) < window:
            return pd.Series(index=close.index, dtype=float)
        
        entropy_vals = []
        for i in range(len(returns) - window + 1):
            window_returns = returns.iloc[i:i+window]
            hist, _ = np.histogram(window_returns, bins=min(10, len(window_returns)), density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist))
            entropy_vals.append(entropy)
        
        entropy_series = pd.Series(entropy_vals, index=returns.index[window-1:])
        return entropy_series.reindex(close.index, method='ffill')
    
    def entropic_memory_break(close, volume, window=5):
        """Calculate entropic memory break using volume-weighted price changes"""
        price_change = close.diff()
        volume_ratio = volume / volume.shift(1)
        memory_break = (price_change * volume_ratio).rolling(window=window).std()
        return memory_break
    
    def fractal_memory_hold(close, volume, window=5):
        """Calculate fractal memory hold using autocorrelation of volume-scaled returns"""
        volume_scaled_returns = (close.diff() / close.shift(1)) * volume
        autocorr = volume_scaled_returns.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        return autocorr
    
    def microstructure_fractal(high, low, close, volume, window=5):
        """Calculate microstructure fractal combining price and volume patterns"""
        price_range = (high - low) / ((high + low) / 2)
        volume_change = volume.diff()
        micro_fractal = (price_range * volume_change).rolling(window=window).mean()
        return micro_fractal
    
    # Calculate base components
    df['fractal_dim'] = price_fractal_dimension(df['high'], df['low'], df['close'])
    df['price_entropy'] = price_entropy(df['close'])
    df['entropic_memory'] = entropic_memory_break(df['close'], df['volume'])
    df['fractal_memory'] = fractal_memory_hold(df['close'], df['volume'])
    df['micro_fractal'] = microstructure_fractal(df['high'], df['low'], df['close'], df['volume'])
    
    # Fractal Microstructure Efficiency
    df['fractal_spread_momentum'] = (
        (df['close'] - df['close'].shift(1)) * 
        (df['high'] - df['low']) / ((df['high'] + df['low']) / 2) * 
        df['fractal_dim']
    )
    
    df['entropic_gap_efficiency'] = (
        (df['close'] - df['open']) * 
        np.abs(df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8) * 
        df['price_entropy']
    )
    
    df['memory_impact_efficiency'] = (
        np.abs(df['close'] - df['close'].shift(1)) / (df['volume'] + 1e-8) * 
        (df['volume'] / (df['volume'].shift(1) + 1e-8)) * 
        df['entropic_memory']
    )
    
    # Quantum Volume-Memory Dynamics
    df['memory_volume_burst'] = (
        (df['close'] - df['close'].shift(1)) * 
        (df['volume'] / (df['volume'].shift(1) + 1e-8) - df['volume'].shift(1) / (df['volume'].shift(2) + 1e-8)) * 
        df['fractal_memory']
    )
    
    df['entropic_acceleration_divergence'] = (
        (df['close'] - df['close'].shift(1)) * 
        np.sign(df['volume'] / (df['volume'].shift(1) + 1e-8) - df['volume'].shift(1) / (df['volume'].shift(2) + 1e-8)) * 
        df['entropic_memory']
    )
    
    df['quantum_memory_coupling'] = (
        (df['close'] - df['close'].shift(1)) * 
        (df['volume'] / (df['volume'].shift(1) + 1e-8)) ** 2 * 
        (df['volume'] / (df['volume'].shift(1) + 1e-8)) ** (-2) * 
        df['micro_fractal']
    )
    
    # Memory-Regime Transition Framework
    df['fractal_volatility_momentum'] = (
        (df['close'] - df['close'].shift(1)) * 
        (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) * 
        df['fractal_dim']
    )
    
    df['memory_liquidity_transition'] = (
        (df['close'] - df['close'].shift(1)) * 
        np.sign(df['volume'] - df['volume'].shift(1)) * 
        np.sign(df['volume'].shift(1) - df['volume'].shift(2)) * 
        df['fractal_memory']
    )
    
    df['quantum_regime_alignment'] = (
        np.sign((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) - 1) * 
        np.sign(df['volume'] - df['volume'].shift(1)) * 
        df['micro_fractal']
    )
    
    # Quantum Path Asymmetry Memory
    df['memory_efficiency_momentum'] = (
        np.sign(df['close'] - df['open']) * 
        (df['high'] - df['low']) / (np.abs(df['close'] - df['open']) + 1e-8) * 
        (df['close'] - df['close'].shift(1)) * 
        df['entropic_memory']
    )
    
    df['fractal_path_strength'] = (
        ((df['high'] - df['open']) - (df['open'] - df['low'])) / (df['high'] - df['low'] + 1e-8) * 
        df['fractal_dim']
    )
    
    df['quantum_path_quality'] = (
        df['memory_efficiency_momentum'] * 
        df['fractal_path_strength'] * 
        df['micro_fractal']
    )
    
    # Memory Divergence Validation
    df['volume_memory_divergence'] = (
        np.sign(df['memory_volume_burst']) * 
        np.sign(df['entropic_acceleration_divergence'])
    )
    
    df['regime_memory_alignment'] = (
        np.sign(df['fractal_volatility_momentum']) * 
        np.sign(df['memory_liquidity_transition'])
    )
    
    df['path_memory_coherence'] = (
        np.sign(df['memory_efficiency_momentum']) * 
        np.sign(df['quantum_path_quality'])
    )
    
    # Quantum Memory Persistence Framework
    def count_sign_persistence(series, window=3):
        """Count sign persistence over rolling window"""
        sign_series = np.sign(series)
        persistence = pd.Series(index=series.index, dtype=float)
        
        for i in range(window-1, len(series)):
            window_data = sign_series.iloc[i-window+1:i+1]
            if len(window_data) < window:
                persistence.iloc[i] = 0
                continue
            count = sum(window_data.iloc[j] == window_data.iloc[j-1] for j in range(1, len(window_data)))
            persistence.iloc[i] = count / (window - 1)
        
        return persistence
    
    df['volume_memory_persistence'] = count_sign_persistence(df['memory_volume_burst'])
    df['regime_memory_consistency'] = count_sign_persistence(df['quantum_regime_alignment'])
    df['path_memory_stability'] = count_sign_persistence(df['quantum_path_quality'])
    
    # Core Memory Components
    df['fractal_efficiency_core'] = (
        df['fractal_spread_momentum'] * 
        df['entropic_gap_efficiency'] * 
        df['memory_impact_efficiency']
    )
    
    df['quantum_volume_core'] = (
        df['memory_volume_burst'] * 
        df['entropic_acceleration_divergence'] * 
        df['quantum_memory_coupling']
    )
    
    df['memory_regime_core'] = (
        df['fractal_volatility_momentum'] * 
        df['memory_liquidity_transition'] * 
        df['quantum_regime_alignment']
    )
    
    # Validated Memory Integration
    df['divergence_enhanced_core'] = (
        df['fractal_efficiency_core'] * 
        df['volume_memory_divergence']
    )
    
    df['regime_confirmed_volume'] = (
        df['quantum_volume_core'] * 
        df['regime_memory_alignment']
    )
    
    df['path_stable_memory'] = (
        df['quantum_path_quality'] * 
        df['path_memory_stability']
    )
    
    # Persistence-Weighted Components
    df['volume_persistent_memory'] = (
        df['divergence_enhanced_core'] * 
        df['volume_memory_persistence']
    )
    
    df['regime_consistent_memory'] = (
        df['regime_confirmed_volume'] * 
        df['regime_memory_consistency']
    )
    
    df['path_stable_momentum'] = (
        df['path_stable_memory'] * 
        df['path_memory_stability']
    )
    
    # Final Quantum Microstructure Alpha
    primary_factor = df['volume_persistent_memory'] * df['path_memory_coherence']
    secondary_factor = df['regime_consistent_memory'] * df['volume_memory_divergence']
    tertiary_factor = df['path_stable_momentum'] * df['regime_memory_alignment']
    
    # Composite Alpha with persistence validation and coherence alignment
    composite_alpha = (
        0.5 * primary_factor + 
        0.3 * secondary_factor + 
        0.2 * tertiary_factor
    )
    
    # Clean up and return
    alpha_series = composite_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
