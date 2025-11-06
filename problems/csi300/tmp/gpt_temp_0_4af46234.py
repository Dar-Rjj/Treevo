import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Micro-Structure Momentum Divergence factor
    Combines multi-scale fractal analysis with micro-structure order flow
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Helper function for fractal dimension approximation using Higuchi method
    def fractal_dimension(series, k_max=5):
        """Calculate approximate fractal dimension using Higuchi method"""
        if len(series) < k_max:
            return np.nan
            
        L = []
        N = len(series)
        
        for k in range(1, k_max + 1):
            Lk = 0
            for m in range(k):
                # Calculate length for each lag
                idx = np.arange(m, N, k)
                if len(idx) > 1:
                    Lkm = np.sum(np.abs(np.diff(series.iloc[idx])))
                    Lkm = Lkm * (N - 1) / (len(idx) - 1) / k
                    Lk += Lkm
            L.append(np.log(Lk / k))
        
        if len(L) > 1:
            x = np.log(np.arange(1, k_max + 1))
            return np.polyfit(x, L, 1)[0]  # Slope gives fractal dimension
        return np.nan
    
    # Calculate price fractal dimensions
    price_3d_fd = data['close'].rolling(window=3, min_periods=3).apply(
        lambda x: fractal_dimension(x), raw=False
    )
    price_5d_fd = data['close'].rolling(window=5, min_periods=5).apply(
        lambda x: fractal_dimension(x), raw=False
    )
    
    # Calculate volume fractal dimensions
    volume_3d_fd = data['volume'].rolling(window=3, min_periods=3).apply(
        lambda x: fractal_dimension(x), raw=False
    )
    
    # Price fractal momentum components
    price_fractal_momentum_3d = price_3d_fd.diff(1)
    price_fractal_momentum_5d = price_5d_fd.diff(1)
    
    # Volume fractal dynamics
    volume_fractal_momentum = volume_3d_fd.diff(1)
    volume_complexity_accel = volume_fractal_momentum.diff(1)
    
    # Micro-structure order flow components
    price_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    efficiency_momentum = price_efficiency.diff(1)
    
    # Fractal compression-expansion cycles
    low_fractal_phase = (price_3d_fd < price_3d_fd.rolling(window=10).mean()).astype(int)
    fractal_breakout = (price_3d_fd > price_3d_fd.rolling(window=10).mean() * 1.1).astype(int)
    
    # Volume-volatility fractal confirmation
    volume_complexity_roc = volume_3d_fd.pct_change(3)
    volume_complexity_vs_avg = volume_3d_fd / volume_3d_fd.rolling(window=10).mean()
    
    # Daily range fractal dimension
    daily_range = data['high'] - data['low']
    range_fractal = daily_range.rolling(window=5, min_periods=5).apply(
        lambda x: fractal_dimension(x), raw=False
    )
    
    # Volatility-fractal interaction
    volatility_complexity = range_fractal
    volume_accel_vol_complexity = volume_complexity_accel * volatility_complexity
    
    # Adaptive fractal signals
    # High compression strategy
    compression_level = price_3d_fd.rolling(window=10).rank(pct=True)
    expansion_anticipation = (compression_level < 0.3).astype(int)
    compression_release = (compression_level.diff(1) > 0.1).astype(int)
    
    # Normal fractal strategy
    combined_fractal = (price_3d_fd * 0.6 + volume_3d_fd * 0.4)
    fractal_cycle = combined_fractal.rolling(window=5).mean()
    
    # High complexity strategy
    extreme_complexity = (price_3d_fd > price_3d_fd.rolling(window=20).quantile(0.8)).astype(int)
    contrarian_signal = -extreme_complexity
    
    # Combine all components with weights
    factor = (
        # Multi-scale fractal momentum (30%)
        price_fractal_momentum_3d * 0.15 +
        price_fractal_momentum_5d * 0.15 +
        
        # Volume fractal dynamics (20%)
        volume_fractal_momentum * 0.1 +
        volume_complexity_accel * 0.1 +
        
        # Micro-structure order flow (20%)
        price_efficiency * 0.1 +
        efficiency_momentum * 0.1 +
        
        # Volume-volatility confirmation (15%)
        volume_complexity_roc * 0.05 +
        volume_complexity_vs_avg * 0.05 +
        volume_accel_vol_complexity * 0.05 +
        
        # Adaptive signals (15%)
        expansion_anticipation * 0.05 +
        compression_release * 0.05 +
        contrarian_signal * 0.05
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20).mean()) / (factor.rolling(window=20).std() + 1e-8)
    
    return factor
