import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum blend
    momentum_short = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_medium = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    momentum_long = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)
    
    # Volatility-scaled volume
    volume_volatility = df['volume'].rolling(window=20, min_periods=10).std()
    price_volatility = df['close'].rolling(window=20, min_periods=10).std() / df['close'].shift(20)
    volatility_scaled_volume = df['volume'] / (volume_volatility * price_volatility + 1e-7)
    
    # Quantile-based regime detection
    range_ratio = (df['high'] - df['low']) / df['close']
    volume_trend = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    
    range_regime = range_ratio.rolling(window=20, min_periods=10).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else (-1 if x.iloc[-1] < x.quantile(0.3) else 0)
    )
    volume_regime = volume_trend.rolling(window=20, min_periods=10).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else (-1 if x.iloc[-1] < x.quantile(0.3) else 0)
    )
    
    regime_multiplier = 1 + 0.2 * (range_regime + volume_regime)
    
    # Combine components with multiplicative interactions
    momentum_blend = 0.4 * momentum_short + 0.4 * momentum_medium + 0.2 * momentum_long
    alpha_factor = momentum_blend * volatility_scaled_volume * regime_multiplier
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
