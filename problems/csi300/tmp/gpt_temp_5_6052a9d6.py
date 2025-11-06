import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-Normalized Momentum with Volume-Weighted Decay and Range Efficiency
    Combines multi-timeframe momentum signals normalized by their volatility regimes
    Applies volume-weighted decay to emphasize recent volume confirmation
    Range efficiency captures intraday momentum persistence with decay
    """
    
    # Multi-timeframe momentum calculations
    momentum_2d = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_8d = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    
    # Volatility normalization using rolling standard deviation of returns
    returns = df['close'].pct_change()
    vol_3d = returns.rolling(window=3).std()
    vol_6d = returns.rolling(window=6).std()
    vol_10d = returns.rolling(window=10).std()
    
    # Volatility-normalized momentum signals
    mom_norm_2d = momentum_2d / (vol_3d + 1e-7)
    mom_norm_5d = momentum_5d / (vol_6d + 1e-7)
    mom_norm_8d = momentum_8d / (vol_10d + 1e-7)
    
    # Volume divergence with decay weighting
    volume_avg_2d = df['volume'].rolling(window=2).mean()
    volume_avg_5d = df['volume'].rolling(window=5).mean()
    volume_avg_8d = df['volume'].rolling(window=8).mean()
    
    volume_div_2d = (df['volume'] - volume_avg_2d) / volume_avg_2d
    volume_div_5d = (df['volume'] - volume_avg_5d) / volume_avg_5d
    volume_div_8d = (df['volume'] - volume_avg_8d) / volume_avg_8d
    
    # Volume-weighted decay factors using exponential decay
    vol_weight_2d = volume_div_2d * np.exp(-0.4)
    vol_weight_5d = volume_div_5d * np.exp(-0.25)
    vol_weight_8d = volume_div_8d * np.exp(-0.15)
    
    # Volume-weighted momentum convergence
    momentum_convergence = (
        mom_norm_2d * (1 + vol_weight_2d) +
        mom_norm_5d * (1 + vol_weight_5d) +
        mom_norm_8d * (1 + vol_weight_8d)
    )
    
    # Range efficiency with exponential decay
    true_range = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    range_efficiency = (df['close'] - df['close'].shift(1)) / (true_range + 1e-7)
    
    # Apply exponential decay to range efficiency signal
    decay_weights = np.exp(-0.2 * np.arange(len(df)))
    range_signal = range_efficiency.rolling(window=3).apply(
        lambda x: np.sum(x * decay_weights[:len(x)]) / np.sum(decay_weights[:len(x)])
    )
    
    # Final alpha factor combining momentum convergence and range signals
    alpha = 0.7 * momentum_convergence + 0.3 * range_signal
    
    return alpha
