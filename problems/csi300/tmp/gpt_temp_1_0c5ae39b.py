import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume-Range Efficiency Momentum with Adaptive Regimes
    Combines fractal analysis, quantum-inspired entanglement, biological swarm patterns,
    chaos theory, information theory, and topological data analysis.
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Fractal Market Structure Analysis
    def calculate_fractal_dimension(prices, window):
        """Calculate fractal dimension using Higuchi method approximation"""
        if len(prices) < window:
            return np.nan
        
        L = []
        for k in range(1, min(10, window//2)):
            Lk = 0
            for m in range(k):
                segments = prices[m::k]
                if len(segments) > 1:
                    Lk += np.sum(np.abs(np.diff(segments)))
            Lk = Lk * (window - 1) / (k**2 * (window / k))
            L.append(Lk)
        
        if len(L) > 1:
            x = np.log(np.arange(1, len(L)+1))
            y = np.log(L)
            return -np.polyfit(x, y, 1)[0]
        return np.nan
    
    # Multi-Timeframe Fractal Dimension
    df['fractal_5d'] = df['close'].rolling(window=5, min_periods=5).apply(
        lambda x: calculate_fractal_dimension(x.values, 5), raw=False
    )
    df['fractal_20d'] = df['close'].rolling(window=20, min_periods=20).apply(
        lambda x: calculate_fractal_dimension(x.values, 20), raw=False
    )
    df['fractal_ratio'] = df['fractal_5d'] / df['fractal_20d']
    
    # Volume Fractal Patterns
    df['volume_fractal_5d'] = df['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: calculate_fractal_dimension(x.values, 5), raw=False
    )
    df['volume_clustering'] = df['volume'].rolling(window=10).apply(
        lambda x: np.std(x) / (np.mean(x) + 1e-8), raw=False
    )
    
    # Range Efficiency Fractal Score
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['range_efficiency'] = (df['close'] - df['open']) / (df['true_range'] + 1e-8)
    df['range_fractal'] = df['range_efficiency'].rolling(window=10).apply(
        lambda x: calculate_fractal_dimension(x.values, 10), raw=False
    )
    
    # Quantum-Inspired Price-Volume Entanglement
    df['price_momentum_5'] = df['close'].pct_change(5)
    df['volume_momentum_5'] = df['volume'].pct_change(5)
    df['pv_correlation_10'] = df['price_momentum_5'].rolling(window=10).corr(df['volume_momentum_5'])
    
    # Entanglement persistence
    df['entanglement_persistence'] = df['pv_correlation_10'].rolling(window=5).apply(
        lambda x: np.mean(np.abs(np.diff(x))), raw=False
    )
    
    # Biological Swarm Intelligence Patterns
    df['price_swarm_coherence'] = (
        df['close'].rolling(window=5).std() / 
        (df['close'].rolling(window=20).std() + 1e-8)
    )
    df['volume_swarm_density'] = (
        df['volume'].rolling(window=5).mean() / 
        (df['volume'].rolling(window=20).mean() + 1e-8)
    )
    
    # Chaos Theory - Lyapunov exponent approximation
    def approximate_lyapunov(returns, window):
        """Approximate Lyapunov exponent using divergence rates"""
        if len(returns) < window:
            return np.nan
        divergences = []
        for i in range(len(returns) - window):
            base_traj = returns[i:i+window//2]
            for j in range(i+1, len(returns) - window):
                comp_traj = returns[j:j+window//2]
                if len(base_traj) == len(comp_traj):
                    initial_dist = np.abs(base_traj[0] - comp_traj[0])
                    final_dist = np.abs(base_traj[-1] - comp_traj[-1])
                    if initial_dist > 1e-8:
                        divergences.append(np.log(final_dist / initial_dist))
        return np.mean(divergences) if divergences else 0
    
    df['chaos_lyapunov_10'] = df['close'].pct_change().rolling(window=10).apply(
        lambda x: approximate_lyapunov(x.values, 10), raw=False
    )
    
    # Information Theory - Entropy
    def calculate_entropy(series, bins=10):
        """Calculate Shannon entropy"""
        hist, _ = np.histogram(series.dropna(), bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))
    
    df['price_entropy_10'] = df['close'].pct_change().rolling(window=10).apply(
        lambda x: calculate_entropy(x), raw=False
    )
    df['volume_entropy_10'] = df['volume'].pct_change().rolling(window=10).apply(
        lambda x: calculate_entropy(x), raw=False
    )
    
    # Topological persistence approximation
    df['topological_persistence'] = (
        df['high'].rolling(window=5).max() - 
        df['low'].rolling(window=5).min()
    ) / (df['true_range'].rolling(window=5).mean() + 1e-8)
    
    # Multi-Paradigm Signal Fusion
    factors = []
    
    # Fractal regime component
    fractal_component = (
        df['fractal_ratio'] * 
        df['range_fractal'] * 
        (1 - df['volume_clustering'])
    )
    factors.append(fractal_component)
    
    # Quantum entanglement component
    quantum_component = (
        np.abs(df['pv_correlation_10']) * 
        (1 - df['entanglement_persistence']) *
        df['range_efficiency']
    )
    factors.append(quantum_component)
    
    # Biological swarm component
    swarm_component = (
        df['price_swarm_coherence'] * 
        df['volume_swarm_density'] *
        df['price_momentum_5']
    )
    factors.append(swarm_component)
    
    # Chaos theory component
    chaos_component = (
        np.abs(df['chaos_lyapunov_10']) * 
        df['topological_persistence'] *
        df['price_entropy_10']
    )
    factors.append(chaos_component)
    
    # Information theory component
    info_component = (
        (1 - df['price_entropy_10']) * 
        (1 - df['volume_entropy_10']) *
        df['pv_correlation_10']
    )
    factors.append(info_component)
    
    # Combine all components with adaptive weighting
    valid_factors = [f for f in factors if not f.isnull().all()]
    
    if valid_factors:
        # Z-score normalization for each factor
        z_factors = []
        for factor in valid_factors:
            mean = factor.rolling(window=20, min_periods=10).mean()
            std = factor.rolling(window=20, min_periods=10).std()
            z_factors.append((factor - mean) / (std + 1e-8))
        
        # Equal weighting for simplicity
        combined_factor = sum(z_factors) / len(z_factors)
        
        # Final smoothing and regime adaptation
        result = combined_factor.rolling(window=3).mean()
    
    return result.fillna(0)
