import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Fractal Analysis
    # 10-day price range efficiency (High-Low volatility)
    high_low_range = df['high'] - df['low']
    price_range_efficiency = high_low_range.rolling(window=10).std() / df['close'].rolling(window=10).mean()
    
    # 20-day price path complexity (Close price gyration)
    close_returns = df['close'].pct_change()
    price_path_complexity = close_returns.rolling(window=20).apply(
        lambda x: np.sqrt(np.sum(x**2)) / (np.max(x) - np.min(x)) if (np.max(x) - np.min(x)) != 0 else 0
    )
    
    # Volume Fractal Analysis
    # 15-day volume clustering intensity
    volume_ma = df['volume'].rolling(window=15).mean()
    volume_std = df['volume'].rolling(window=15).std()
    volume_clustering = (df['volume'] - volume_ma) / (volume_std + 1e-8)
    volume_clustering_intensity = volume_clustering.rolling(window=15).apply(
        lambda x: np.sum(np.abs(x) > 1) / len(x) if len(x) > 0 else 0
    )
    
    # Volume burst persistence ratio
    volume_above_ma = (df['volume'] > volume_ma).astype(int)
    volume_burst_persistence = volume_above_ma.rolling(window=15).sum() / 15
    
    # Fractal Convergence Detection
    # Price vs volume fractal trend comparison
    price_trend = df['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    volume_trend = df['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # Convergence/divergence identification
    trend_convergence = np.sign(price_trend) * np.sign(volume_trend)
    convergence_strength = np.abs(price_trend * volume_trend)
    
    # Signal Generation
    # Convergence strength weighting by efficiency
    efficiency_weighted_convergence = trend_convergence * convergence_strength * (1 - price_range_efficiency)
    
    # Fractal stability scaling
    fractal_stability = 1 / (1 + price_path_complexity + volume_clustering_intensity)
    
    # Final factor combining all components
    factor = efficiency_weighted_convergence * fractal_stability * (1 + volume_burst_persistence)
    
    return factor
