import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-Normalized Multi-Timeframe Momentum with Volume Divergence
    # Combines short and medium-term momentum signals with volume confirmation across regimes
    # Interpretable as: Stocks showing persistent momentum across timeframes with volume validation
    
    # Multi-timeframe price momentum (3-day and 10-day)
    mom_short = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_medium = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Volume divergence: volume momentum relative to price momentum
    vol_mom_short = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    vol_mom_medium = (df['volume'] - df['volume'].shift(10)) / df['volume'].shift(10)
    
    # Volume-price divergence signals
    vol_div_short = vol_mom_short - mom_short
    vol_div_medium = vol_mom_medium - mom_medium
    
    # Rolling volatility regimes (5-day and 20-day windows)
    returns = df['close'].pct_change()
    vol_short = returns.rolling(window=5).std()
    vol_medium = returns.rolling(window=20).std()
    
    # Volatility-normalized momentum with regime adaptation
    # Short-term momentum weighted by inverse of medium-term volatility
    mom_norm_short = mom_short / (vol_medium + 1e-7)
    
    # Medium-term momentum with short-term volatility context
    mom_norm_medium = mom_medium / (vol_short + 1e-7)
    
    # Volume divergence signals with volatility adjustment
    vol_div_norm_short = vol_div_short / (vol_short + 1e-7)
    vol_div_norm_medium = vol_div_medium / (vol_medium + 1e-7)
    
    # Decay-adjusted convergence signals using exponential weighting
    # Recent signals get higher weight
    mom_convergence = (
        0.6 * mom_norm_short.rolling(window=5).apply(lambda x: np.average(x, weights=np.exp(np.arange(len(x))))) +
        0.4 * mom_norm_medium.rolling(window=10).apply(lambda x: np.average(x, weights=np.exp(np.arange(len(x)))))
    )
    
    vol_div_convergence = (
        0.7 * vol_div_norm_short.rolling(window=3).apply(lambda x: np.average(x, weights=np.exp(np.arange(len(x))))) +
        0.3 * vol_div_norm_medium.rolling(window=7).apply(lambda x: np.average(x, weights=np.exp(np.arange(len(x)))))
    )
    
    # Final alpha: volatility-normalized momentum with volume divergence confirmation
    alpha = mom_convergence * vol_div_convergence
    
    return alpha
