import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor with volatility-normalized momentum, volume divergence,
    and multi-timeframe geometric combination. Focuses on momentum reversals
    and price efficiency with exponential decay weighting.
    """
    # Volatility-normalized momentum with exponential decay
    decay_weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # 5-day exponential decay
    
    momentum_components = []
    for i, weight in enumerate(decay_weights, 1):
        ret = (df['close'] - df['close'].shift(i)) / df['close'].shift(i)
        momentum_components.append(weight * ret)
    
    raw_momentum = sum(momentum_components)
    
    # 10-day exponential volatility (using close-to-close returns)
    returns = df['close'].pct_change()
    volatility = returns.ewm(span=10).std()
    
    volatility_normalized_momentum = raw_momentum / (volatility + 1e-7)
    
    # Volume divergence with exponential weighting
    volume_ema_10 = df['volume'].ewm(span=10).mean()
    volume_divergence = df['volume'] / (volume_ema_10 + 1e-7)
    
    # Volume acceleration (1-day vs 5-day trend)
    volume_trend = df['volume'].ewm(span=5).mean()
    volume_acceleration = (df['volume'] - volume_trend.shift(1)) / (volume_trend.shift(1) + 1e-7)
    
    # Multi-timeframe geometric combination
    short_term_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    medium_term_momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    geometric_combination = (abs(short_term_momentum) ** 0.5 * np.sign(short_term_momentum) * 
                           medium_term_momentum)
    
    # Momentum reversal detection
    intraday_range = df['high'] - df['low']
    price_efficiency = (df['close'] - df['open']) / (intraday_range + 1e-7)
    
    # Overbought/oversold using rolling percentiles
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_percentile = momentum_5d.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Price efficiency: opening gap persistence
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_persistence = (df['close'] - df['open']) / (abs(opening_gap) * df['close'].shift(1) + 1e-7)
    
    # Combine factors with clear economic interpretation
    factor = (
        volatility_normalized_momentum *
        (1 + 0.15 * volume_divergence) *
        (1 + 0.1 * volume_acceleration) *
        (1 + 0.2 * geometric_combination) *
        (1 - 0.15 * abs(momentum_percentile)) *  # Reversal component
        (1 + 0.1 * price_efficiency) *
        (1 + 0.08 * gap_persistence)
    )
    
    return factor
