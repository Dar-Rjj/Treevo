import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe volatility-aligned alpha factor combining:
    - Price momentum with volatility regime weighting
    - Volume pressure with dynamic timeframe adjustment
    - Range efficiency with volatility scaling
    - Liquidity momentum with regime adaptation
    - Trend persistence with volatility filtering
    """
    
    # Multi-timeframe momentum with volatility regime alignment
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_20d = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Volatility regime detection using aligned timeframes
    returns_10d = df['close'].pct_change(10)
    vol_10d = returns_10d.rolling(window=20, min_periods=12).std()
    vol_20d = returns_10d.rolling(window=40, min_periods=24).std()
    volatility_regime = vol_10d / (vol_20d + 1e-7)
    
    # Dynamic momentum weighting based on volatility regime
    momentum_weight = 1.0 / (1.0 + np.exp(-4 * (volatility_regime - 1.1)))
    momentum_aligned = (
        momentum_weight * momentum_5d + 
        (0.7 - 0.3 * momentum_weight) * momentum_10d + 
        (0.3 - 0.1 * momentum_weight) * momentum_20d
    )
    
    # Multi-timeframe volume pressure with volatility scaling
    volume_pressure_5d = df['volume'] / df['volume'].rolling(window=10, min_periods=6).mean()
    volume_pressure_10d = df['volume'] / df['volume'].rolling(window=20, min_periods=12).mean()
    volume_pressure_20d = df['volume'] / df['volume'].rolling(window=40, min_periods=24).mean()
    
    volume_weight = 1.0 / (1.0 + np.exp(-3 * (volatility_regime - 0.9)))
    volume_pressure_aligned = (
        volume_weight * volume_pressure_5d + 
        (0.8 - 0.4 * volume_weight) * volume_pressure_10d + 
        (0.4 - 0.2 * volume_weight) * volume_pressure_20d
    )
    
    # Range efficiency with multi-timeframe volatility scaling
    daily_range = df['high'] - df['low']
    range_efficiency = np.abs(df['close'] - df['close'].shift(1)) / (daily_range + 1e-7)
    
    # Multi-timeframe range volatility adjustment
    range_vol_5d = range_efficiency.rolling(window=10, min_periods=6).std()
    range_vol_10d = range_efficiency.rolling(window=20, min_periods=12).std()
    range_vol_regime = range_vol_5d / (range_vol_10d + 1e-7)
    
    range_volatility_adj = range_efficiency * (1.0 + 0.5 * np.tanh(2 * (range_vol_regime - 1.0)))
    
    # Multi-timeframe liquidity momentum with regime adaptation
    avg_trade_size = df['amount'] / (df['volume'] + 1e-7)
    liquidity_5d = avg_trade_size / avg_trade_size.rolling(window=10, min_periods=6).mean()
    liquidity_10d = avg_trade_size / avg_trade_size.rolling(window=20, min_periods=12).mean()
    liquidity_20d = avg_trade_size / avg_trade_size.rolling(window=40, min_periods=24).mean()
    
    liquidity_weight = 1.0 / (1.0 + np.exp(-2.5 * (volatility_regime - 0.85)))
    liquidity_aligned = (
        liquidity_weight * liquidity_5d + 
        (0.75 - 0.35 * liquidity_weight) * liquidity_10d + 
        (0.35 - 0.15 * liquidity_weight) * liquidity_20d
    )
    
    # Trend persistence with volatility filtering
    trend_5d = df['close'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan
    )
    trend_10d = df['close'].rolling(window=10, min_periods=6).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 6 else np.nan
    )
    
    trend_weight = 1.0 / (1.0 + np.exp(-3 * (volatility_regime - 1.05)))
    trend_persistence = trend_weight * trend_5d + (1 - trend_weight) * trend_10d
    
    # Multiplicative combination with volatility-aware scaling
    alpha_factor = (
        momentum_aligned * 
        volume_pressure_aligned * 
        range_volatility_adj * 
        liquidity_aligned * 
        (1.0 + 0.3 * np.tanh(trend_persistence / (df['close'] + 1e-7)))
    )
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
