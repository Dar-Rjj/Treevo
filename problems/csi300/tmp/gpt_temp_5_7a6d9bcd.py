import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining momentum divergence, volume-price efficiency,
    volatility-adjusted persistence, and liquidity conditions.
    """
    # Price-Based Momentum & Reversal
    # Multi-timeframe momentum divergence
    mom_5 = df['close'] / df['close'].shift(5) - 1
    mom_20 = df['close'] / df['close'].shift(20) - 1
    momentum_divergence = (mom_5 - mom_20) / (df['close'].rolling(20).std() + 1e-8)
    
    # Momentum acceleration (2nd derivative)
    mom_accel = mom_5.diff(3) / (df['close'].rolling(10).std() + 1e-8)
    
    # Gap-based mean reversion
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_fill_prob = -opening_gap.rolling(10).apply(lambda x: (x[:-1] * x[1:] < 0).mean())
    
    # Volume-Price Relationship
    # Volume-weighted price efficiency
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    price_efficiency = (df['close'] - df['open']) / (true_range + 1e-8)
    volume_weighted_efficiency = price_efficiency * df['volume']
    
    # Volume acceleration confirmation
    vol_accel = df['volume'].pct_change(3).rolling(5).mean()
    volume_confirmation = volume_weighted_efficiency * vol_accel
    
    # Multi-period divergence
    vol_short = df['volume'].rolling(3).mean()
    vol_medium = df['volume'].rolling(10).mean()
    volume_divergence = (vol_short - vol_medium) / (vol_medium + 1e-8)
    price_trend = df['close'].rolling(5).mean() / df['close'].rolling(10).mean() - 1
    trend_alignment = volume_divergence * price_trend
    
    # Volatility & Risk Adjustment
    # Return persistence under volatility
    returns = df['close'].pct_change()
    same_sign_count = returns.rolling(5).apply(lambda x: sum((x[:-1] * x[1:]) > 0))
    volatility = df['close'].pct_change().rolling(10).std()
    volatility_scaled_persistence = same_sign_count / (volatility + 1e-8)
    
    # Liquidity-adjusted signals
    liquidity_score = (df['volume'] * df['amount']).rolling(5).mean()
    liquidity_weight = liquidity_score / liquidity_score.rolling(20).mean()
    
    # Composite Signal Construction
    # Cross-validated sub-factors with non-linear combinations
    momentum_component = np.tanh(momentum_divergence) * np.sign(mom_accel)
    volume_component = np.tanh(volume_confirmation) * trend_alignment
    persistence_component = np.tanh(volatility_scaled_persistence) * gap_fill_prob
    
    # Time-decay weighting for recent signals
    decay_weights = np.exp(-np.arange(5) / 2.0)  # Exponential decay
    recent_momentum = momentum_component.rolling(5).apply(
        lambda x: np.dot(x, decay_weights) / sum(decay_weights)
    )
    
    # Final composite factor
    composite_factor = (
        momentum_component * 0.3 +
        volume_component * 0.25 +
        persistence_component * 0.2 +
        recent_momentum * 0.15 +
        liquidity_weight * 0.1
    )
    
    # Normalize by recent volatility
    factor_volatility = composite_factor.rolling(20).std()
    normalized_factor = composite_factor / (factor_volatility + 1e-8)
    
    return normalized_factor
