import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced multi-timeframe alpha factor with dynamic signal weighting, volatility regime adaptation,
    and cross-timeframe momentum alignment for improved return prediction.
    
    Economic intuition:
    - Multi-timeframe momentum captures trend persistence across different horizons
    - Dynamic weighting emphasizes stronger and more persistent signals
    - Volatility regime adaptation adjusts factor sensitivity to market conditions
    - Cross-timeframe alignment identifies robust directional moves
    - Volume confirmation enhances signal reliability
    - Range efficiency measures intraday price momentum strength
    """
    
    # Multi-timeframe price momentum (Fibonacci sequence: 1, 3, 5, 8, 13 days)
    momentum_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_8d = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    momentum_13d = (df['close'] - df['close'].shift(13)) / df['close'].shift(13)
    
    # Multi-timeframe volume momentum for confirmation
    volume_momentum_1d = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    volume_momentum_3d = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    volume_momentum_5d = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    volume_momentum_8d = (df['volume'] - df['volume'].shift(8)) / df['volume'].shift(8)
    
    # Price efficiency and market structure measures
    daily_range = df['high'] - df['low']
    range_efficiency = (df['close'] - df['low']) / (daily_range + 1e-7)
    open_to_close = (df['close'] - df['open']) / df['open']
    high_to_close = (df['high'] - df['close']) / df['close']
    
    # Volume efficiency and liquidity quality
    volume_efficiency = df['amount'] / (df['volume'] + 1e-7)
    volume_efficiency_momentum = volume_efficiency / volume_efficiency.shift(1) - 1
    
    # Multi-window volatility regime detection
    returns = df['close'].pct_change()
    volatility_3d = returns.rolling(window=3, min_periods=2).std()
    volatility_8d = returns.rolling(window=8, min_periods=6).std()
    volatility_21d = returns.rolling(window=21, min_periods=15).std()
    volatility_regime = volatility_3d / (volatility_21d + 1e-7)
    volatility_trend = volatility_8d / (volatility_21d + 1e-7)
    
    # Enhanced component building with volume confirmation
    short_term = momentum_1d * (1 + volume_momentum_1d) * range_efficiency
    medium_short = momentum_3d * (1 + volume_momentum_3d) * open_to_close
    medium_term = momentum_5d * (1 + volume_momentum_5d) * volume_efficiency_momentum
    medium_long = momentum_8d * (1 + volume_momentum_8d) * high_to_close
    long_term = momentum_13d * (1 + volume_momentum_8d)  # Use 8d volume for long-term
    
    # Cross-timeframe momentum alignment score
    alignment_score = (
        (momentum_1d * momentum_3d > 0).astype(int) +
        (momentum_3d * momentum_5d > 0).astype(int) +
        (momentum_5d * momentum_8d > 0).astype(int) +
        (momentum_8d * momentum_13d > 0).astype(int)
    ) / 4.0
    
    # Dynamic signal persistence and strength assessment
    signal_persistence_short = short_term.rolling(window=3).apply(lambda x: (x > 0).sum() / len(x) if len(x) == 3 else 0.5)
    signal_persistence_medium_short = medium_short.rolling(window=5).apply(lambda x: (x > 0).sum() / len(x) if len(x) == 5 else 0.5)
    signal_persistence_medium = medium_term.rolling(window=8).apply(lambda x: (x > 0).sum() / len(x) if len(x) == 8 else 0.5)
    signal_persistence_medium_long = medium_long.rolling(window=13).apply(lambda x: (x > 0).sum() / len(x) if len(x) == 13 else 0.5)
    signal_persistence_long = long_term.rolling(window=21).apply(lambda x: (x > 0).sum() / len(x) if len(x) == 21 else 0.5)
    
    signal_strength_short = abs(short_term.rolling(window=3).mean()) * signal_persistence_short
    signal_strength_medium_short = abs(medium_short.rolling(window=5).mean()) * signal_persistence_medium_short
    signal_strength_medium = abs(medium_term.rolling(window=8).mean()) * signal_persistence_medium
    signal_strength_medium_long = abs(medium_long.rolling(window=13).mean()) * signal_persistence_medium_long
    signal_strength_long = abs(long_term.rolling(window=21).mean()) * signal_persistence_long
    
    # Enhanced adaptive weighting with persistence and alignment
    total_signal_strength = (
        signal_strength_short + signal_strength_medium_short + 
        signal_strength_medium + signal_strength_medium_long + 
        signal_strength_long + 1e-7
    )
    
    weight_short = (signal_strength_short / total_signal_strength) * (1 + alignment_score) * signal_persistence_short
    weight_medium_short = (signal_strength_medium_short / total_signal_strength) * (1 + alignment_score) * signal_persistence_medium_short
    weight_medium = (signal_strength_medium / total_signal_strength) * (1 + alignment_score) * signal_persistence_medium
    weight_medium_long = (signal_strength_medium_long / total_signal_strength) * (1 + alignment_score) * signal_persistence_medium_long
    weight_long = (signal_strength_long / total_signal_strength) * (1 + alignment_score) * signal_persistence_long
    
    # Hierarchical blend with enhanced dynamic weighting
    hierarchical_blend = (
        short_term * weight_short +
        medium_short * weight_medium_short +
        medium_term * weight_medium +
        medium_long * weight_medium_long +
        long_term * weight_long
    )
    
    # Volatility-adaptive scaling with regime consideration
    volatility_adjustment = (volatility_regime * volatility_trend) + 1e-7
    volatility_sensitivity = 1.0 / (1.0 + volatility_regime)  # Reduce sensitivity in high vol regimes
    
    # Final factor with enhanced volatility adaptation
    factor = hierarchical_blend * volatility_sensitivity / volatility_adjustment
    
    return factor
