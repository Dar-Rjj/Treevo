import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Multi-Scale Price-Volume Divergence
    def linear_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    # Calculate price trends
    price_trend_3d = linear_slope(df['close'], 3)
    price_trend_10d = linear_slope(df['close'], 10)
    price_trend_30d = linear_slope(df['close'], 30)
    
    # Calculate volume trends
    volume_trend_3d = linear_slope(df['volume'], 3)
    volume_trend_10d = linear_slope(df['volume'], 10)
    volume_trend_30d = linear_slope(df['volume'], 30)
    
    # Compare trend directions
    price_volume_alignment_3d = np.sign(price_trend_3d) * np.sign(volume_trend_3d)
    price_volume_alignment_10d = np.sign(price_trend_10d) * np.sign(volume_trend_10d)
    price_volume_alignment_30d = np.sign(price_trend_30d) * np.sign(volume_trend_30d)
    
    # Generate divergence score
    divergence_score = (
        0.5 * price_volume_alignment_3d + 
        0.3 * price_volume_alignment_10d + 
        0.2 * price_volume_alignment_30d
    )
    
    # Volatility-Regime Momentum
    volatility_20d = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    volatility_regime = (volatility_20d > volatility_20d.rolling(window=60).median()).astype(int)
    
    momentum_5d = df['close'].pct_change(5)
    momentum_10d = df['close'].pct_change(10)
    
    regime_weighted_momentum = (
        volatility_regime * momentum_5d + 
        (1 - volatility_regime) * momentum_10d
    )
    
    # Asymmetric Price Impact
    upside_efficiency = (df['high'] - df['open']) / (df['volume'] + 1e-8)
    downside_efficiency = (df['open'] - df['low']) / (df['volume'] + 1e-8)
    
    efficiency_imbalance = upside_efficiency.rolling(window=10).mean() - downside_efficiency.rolling(window=10).mean()
    normalized_imbalance = (efficiency_imbalance - efficiency_imbalance.rolling(window=30).mean()) / (efficiency_imbalance.rolling(window=30).std() + 1e-8)
    
    # Range Compression Energy
    daily_range = df['high'] - df['low']
    range_contraction = (daily_range / daily_range.rolling(window=5).mean()).rolling(window=5).mean()
    
    compression_duration = range_contraction.rolling(window=10).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] < x.iloc[i-1]]), raw=False
    )
    
    compression_energy = range_contraction * compression_duration
    
    # Volume-Driven Momentum Spread
    momentum_3d = df['close'].pct_change(3)
    momentum_5d = df['close'].pct_change(5)
    momentum_10d = df['close'].pct_change(10)
    
    momentum_dispersion = (
        (momentum_3d - momentum_5d).abs() + 
        (momentum_5d - momentum_10d).abs() + 
        (momentum_3d - momentum_10d).abs()
    ) / 3
    
    volume_momentum_alignment = (
        (momentum_3d * df['volume'].pct_change(3)) + 
        (momentum_5d * df['volume'].pct_change(5)) + 
        (momentum_10d * df['volume'].pct_change(10))
    ) / 3
    
    momentum_continuation = momentum_dispersion * volume_momentum_alignment
    
    # Combine all factors
    combined_factor = (
        0.25 * divergence_score + 
        0.25 * regime_weighted_momentum + 
        0.20 * normalized_imbalance + 
        0.15 * compression_energy + 
        0.15 * momentum_continuation
    )
    
    return combined_factor
