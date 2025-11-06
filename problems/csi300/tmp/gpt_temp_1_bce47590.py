import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Enhanced momentum with volume confirmation and volatility dampening
    # Captures trend strength validated by volume and adjusted for market noise
    
    # 1. Dual timeframe momentum (short-term and medium-term)
    momentum_short = df['close'] / df['close'].shift(3) - 1
    momentum_medium = df['close'] / df['close'].shift(8) - 1
    
    # 2. Volume momentum and acceleration
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    volume_acceleration = volume_momentum - volume_momentum.shift(3)
    
    # 3. Volatility regime adjustment using rolling percentiles
    daily_range = (df['high'] - df['low']) / df['close']
    volatility_regime = daily_range.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-7))
    
    # 4. Price-volume efficiency: how much price moves per unit volume
    price_volume_efficiency = (df['close'] - df['open']) / (df['volume'] + 1e-7)
    
    # Combine: momentum alignment + volume confirmation - volatility noise + efficiency bonus
    alpha_factor = (
        (momentum_short * momentum_medium) *  # Aligned momentum across timeframes
        (1 + volume_momentum + volume_acceleration) *  # Volume confirmation and acceleration
        (1 - volatility_regime.clip(0, 1)) *  # Dampen in high volatility regimes
        (1 + price_volume_efficiency.rolling(window=5).mean())  # Recent efficiency bonus
    )
    
    return alpha_factor
