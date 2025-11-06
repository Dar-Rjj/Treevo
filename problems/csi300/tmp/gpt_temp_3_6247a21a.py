import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Intraday Efficiency-Momentum Coupling
    # Calculate efficiency ratio
    efficiency_ratio = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Compute momentum consistency across multiple horizons
    momentum_1d = df['close'].pct_change(1)
    momentum_3d = df['close'].pct_change(3)
    momentum_5d = df['close'].pct_change(5)
    
    # Momentum consistency as correlation between different horizons
    momentum_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        if i >= 20:
            window_data = pd.DataFrame({
                'mom1': momentum_1d.iloc[i-19:i+1],
                'mom3': momentum_3d.iloc[i-19:i+1],
                'mom5': momentum_5d.iloc[i-19:i+1]
            }).corr()
            momentum_consistency.iloc[i] = (window_data.loc['mom1', 'mom3'] + 
                                          window_data.loc['mom1', 'mom5'] + 
                                          window_data.loc['mom3', 'mom5']) / 3
    
    # Detect structural breaks in efficiency patterns using rolling statistics
    efficiency_ma_short = efficiency_ratio.rolling(window=5).mean()
    efficiency_ma_long = efficiency_ratio.rolling(window=20).mean()
    efficiency_break = (efficiency_ma_short - efficiency_ma_long) / (efficiency_ma_long.rolling(window=20).std() + 1e-8)
    
    # Weight momentum by efficiency regime strength
    efficiency_weight = 1 / (1 + np.exp(-3 * efficiency_break))
    weighted_momentum = momentum_5d * efficiency_weight
    
    # Topological Liquidity Flow
    # Estimate buying pressure
    buying_pressure = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Track cumulative volume imbalance with decay
    volume_imbalance = (df['volume'] * buying_pressure - df['volume'] * (1 - buying_pressure))
    decay_factor = 0.95
    cumulative_imbalance = pd.Series(index=df.index, dtype=float)
    cumulative_imbalance.iloc[0] = volume_imbalance.iloc[0]
    for i in range(1, len(df)):
        cumulative_imbalance.iloc[i] = (decay_factor * cumulative_imbalance.iloc[i-1] + 
                                      volume_imbalance.iloc[i])
    
    # Calculate persistent homology from price embeddings (simplified)
    price_embedding = pd.DataFrame({
        'price': df['close'],
        'volume_norm': df['volume'] / df['volume'].rolling(window=20).mean(),
        'range_norm': (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=20).mean()
    }).fillna(0)
    
    # Simplified persistence as rolling correlation stability
    persistence_score = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        corr_matrix = price_embedding.iloc[i-19:i+1].corr()
        persistence_score.iloc[i] = np.linalg.norm(corr_matrix - np.eye(3))
    
    # Weight flow signals by structural persistence
    persistence_weight = 1 / (1 + persistence_score)
    weighted_flow = cumulative_imbalance * persistence_weight
    
    # Fractal Compression Dynamics
    # Compute Hurst exponent across timeframes (simplified)
    def compute_hurst(series, window=20):
        hurst_values = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < 2:
                hurst_values.iloc[i] = 0.5
                continue
            
            lags = range(2, min(10, len(window_data)))
            tau = [np.std(np.subtract(window_data[lag:].values, window_data[:-lag].values)) 
                   for lag in lags]
            hurst_values.iloc[i] = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return hurst_values.fillna(0.5)
    
    hurst_20d = compute_hurst(df['close'], 20)
    
    # Calculate price range contraction ratios
    range_ratio = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=20).mean()
    contraction_ratio = 1 / (1 + range_ratio)
    
    # Compare breakout volume to compression volume
    volume_breakout = df['volume'] / df['volume'].rolling(window=20).mean()
    compression_signal = contraction_ratio * volume_breakout
    
    # Weight signals by regime consistency
    regime_consistency = 1 - np.abs(hurst_20d - 0.5)
    weighted_compression = compression_signal * regime_consistency
    
    # Combine all components with equal weights
    factor = (0.4 * weighted_momentum.fillna(0) + 
              0.3 * weighted_flow.fillna(0) + 
              0.3 * weighted_compression.fillna(0))
    
    return factor
