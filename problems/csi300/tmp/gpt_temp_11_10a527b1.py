import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor combining momentum persistence, volume confirmation, and volatility efficiency.
    Uses relative comparisons, interaction terms, and robust signal combinations for predictive power.
    """
    
    # Momentum persistence: compare short-term vs medium-term momentum
    mom_1d = df['close'].pct_change(1)
    mom_3d = df['close'].pct_change(3)
    momentum_persistence = mom_1d - mom_3d
    
    # Volume confirmation: volume change relative to price momentum
    volume_change = df['volume'].pct_change(1)
    volume_confirmation = volume_change * np.sign(mom_1d)
    
    # Volatility efficiency: close position efficiency relative to recent volatility
    daily_range_pct = (df['high'] - df['low']) / df['close']
    close_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    volatility_efficiency = close_position / (daily_range_pct + 1e-7)
    
    # Relative comparisons using rolling percentiles
    momentum_rank = momentum_persistence.rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-7)
    )
    
    volume_rank = volume_confirmation.rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-7)
    )
    
    volatility_rank = volatility_efficiency.rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-7)
    )
    
    # Interaction terms combining momentum and volume confirmation
    momentum_volume_interaction = momentum_rank * volume_rank
    
    # Robust combination with conditional weighting
    alpha_factor = (
        momentum_rank * 0.4 + 
        volume_rank * 0.3 + 
        volatility_rank * 0.2 + 
        momentum_volume_interaction * 0.1
    )
    
    # Apply bounded nonlinear transform for final signal
    alpha_factor = np.tanh(alpha_factor * 2)
    
    return alpha_factor
