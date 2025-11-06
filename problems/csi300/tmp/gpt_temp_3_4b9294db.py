import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Volatility-Regime Adaptive Price-Volume Convergence
    
    This factor enhances ultra-short term signals by incorporating volatility regime detection
    and multiplicative EMA combinations across price, volume, and amount dimensions:
    1. Price convergence: 1-period EMA vs 2-period EMA divergence for immediate reversals
    2. Volume-amount synergy: Combined flow acceleration using multiplicative EMA
    3. Volatility regime adaptation: Dynamic scaling based on rolling volatility quintiles
    4. Multiplicative EMA alignment: Cross-dimensional smoothing for cleaner signals
    
    Interpretation:
    - Positive values: Price convergence with aligned volume-amount flow in low volatility
    - Negative values: Price divergence with conflicting volume-amount flow in high volatility
    - Magnitude reflects convergence strength adjusted for volatility regime
    
    Economic rationale:
    - EMA combinations reduce noise while preserving ultra-short term sensitivity
    - Volume-amount synergy captures comprehensive liquidity dynamics
    - Volatility quintile scaling adapts signal strength to market conditions
    - Multiplicative structure amplifies aligned cross-dimensional patterns
    - Focus on immediate reversals for ultra-short term mean reversion
    """
    
    # Price convergence: Ultra-short term EMA divergence (1-period vs 2-period)
    ema_1 = df['close'].ewm(span=1, adjust=False).mean()
    ema_2 = df['close'].ewm(span=2, adjust=False).mean()
    price_convergence = (ema_1 - ema_2) / df['close']
    
    # Volume-amount synergy: Multiplicative EMA acceleration
    volume_ema = df['volume'].ewm(span=2, adjust=False).mean()
    amount_ema = df['amount'].ewm(span=2, adjust=False).mean()
    volume_accel = df['volume'] / (volume_ema + 1e-7)
    amount_accel = df['amount'] / (amount_ema + 1e-7)
    liquidity_flow = volume_accel * amount_accel
    
    # Volatility regime detection using rolling quintiles
    daily_range = (df['high'] - df['low']) / df['close']
    vol_regime = daily_range.rolling(window=10, min_periods=5).apply(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop').iloc[-1] if len(x) >= 5 else 2, 
        raw=False
    ).fillna(2)
    
    # Volatility regime scaling factors (inverse relationship)
    regime_scale = 1.0 / (vol_regime + 1.0)
    
    # Multiplicative combination with regime adaptation
    alpha_factor = price_convergence * liquidity_flow * regime_scale
    
    return alpha_factor
