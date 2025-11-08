import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Convergence factor
    """
    # Initialize output series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Multi-Timeframe Momentum Analysis
    # Intraday Momentum Strength
    intraday_return = df['close'] - df['open']
    intraday_range = df['high'] - df['low'] + epsilon
    range_normalized_momentum = intraday_return / intraday_range
    
    # Short-term Momentum (3-day)
    short_term_momentum = df['close'] / df['close'].shift(3) - 1
    # Direction consistency: count positive intraday moves in last 3 days
    intraday_positive = (intraday_return > 0).astype(int)
    direction_consistency = intraday_positive.rolling(window=3, min_periods=1).sum()
    
    # Medium-term Momentum (10-day)
    medium_term_momentum = df['close'] / df['close'].shift(10) - 1
    # Trend persistence: rolling sum of intraday momentum signs
    trend_persistence = np.sign(intraday_return).rolling(window=10, min_periods=1).sum()
    
    # Volume-Price Convergence Signals
    # Volume Momentum Analysis
    volume_change_ratio = df['volume'] / (df['volume'].shift(1) + epsilon)
    volume_trend = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    volume_acceleration = (df['volume'] / (df['volume'].shift(1) + epsilon)) / \
                         (df['volume'].shift(1) / (df['volume'].shift(2) + epsilon) + epsilon)
    
    # Volume-Price Alignment
    price_change = df['close'] / df['close'].shift(1) - 1
    direction_alignment = np.sign(volume_change_ratio - 1) * np.sign(price_change)
    strength_alignment = np.abs(volume_change_ratio - 1) * np.abs(price_change)
    # Convergence persistence: rolling count of aligned days
    convergence_persistence = (direction_alignment > 0).rolling(window=5, min_periods=1).sum()
    
    # Volume Breakout Detection
    volume_spike = df['volume'] > 2 * df['volume'].rolling(window=20, min_periods=1).mean().shift(1)
    price_confirmation = (np.sign(intraday_return) == np.sign(volume_change_ratio - 1)).astype(int)
    breakout_strength = (df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean().shift(1)) * \
                       np.abs(intraday_return / intraday_range)
    
    # Volatility-Scaled Signal Processing
    # Adaptive Volatility Measures
    daily_range_vol = df['high'] - df['low']
    short_term_vol = daily_range_vol.rolling(window=5, min_periods=1).mean()
    volatility_regime = daily_range_vol / (daily_range_vol.rolling(window=20, min_periods=1).mean() + epsilon)
    
    # Volatility-Weighted Signals
    momentum_vol_scaled = range_normalized_momentum / (short_term_vol + epsilon)
    volume_signals_weighted = (volume_change_ratio - 1) * volatility_regime
    
    # Regime-Adaptive Decay
    vol_percentile = daily_range_vol.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)).astype(int), raw=False
    )
    decay_factor = 0.95 - 0.15 * vol_percentile  # 0.8 for high vol, 0.95 for low vol
    
    # Factor Integration & Enhancement
    # Signal Convergence Scoring
    momentum_volume_alignment = direction_alignment * strength_alignment
    multi_timeframe_consistency = (np.sign(short_term_momentum) == np.sign(medium_term_momentum)).astype(int) * \
                                 direction_consistency / 3.0
    volatility_adjusted_strength = momentum_vol_scaled * (1 + 0.5 * (volatility_regime > 1).astype(int))
    
    # Intraday Pattern Recognition
    # Simplified: emphasize morning strength (first hour momentum)
    morning_strength = range_normalized_momentum.rolling(window=5, min_periods=1).apply(
        lambda x: x.iloc[-1] if len(x) > 0 else 0, raw=False
    )
    
    # Final Alpha Construction
    # Base factor components
    momentum_component = 0.4 * range_normalized_momentum + 0.3 * short_term_momentum + 0.3 * medium_term_momentum
    volume_component = 0.3 * volume_change_ratio + 0.4 * volume_trend + 0.3 * volume_acceleration
    alignment_component = 0.5 * momentum_volume_alignment + 0.3 * convergence_persistence + 0.2 * breakout_strength
    
    # Combine with regime adaptation
    base_factor = (0.4 * momentum_component + 0.4 * volume_component + 0.2 * alignment_component) * \
                 volatility_adjusted_strength
    
    # Apply regime-specific scaling and decay
    alpha = base_factor * decay_factor
    
    # Volatility normalization
    alpha_vol = alpha.rolling(window=20, min_periods=1).std() + epsilon
    alpha = (alpha - alpha.rolling(window=20, min_periods=1).mean()) / alpha_vol
    
    return alpha.fillna(0)
