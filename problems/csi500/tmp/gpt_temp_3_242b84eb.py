import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Multi-Timeframe Trend Convergence with Volume Extremes and Volatility Regime Filtering
    
    Factor interpretation:
    - Combines short-term (5-day), medium-term (20-day), and long-term (60-day) momentum to capture trend convergence
    - Volume extreme detection using percentile ranking over 30-day window identifies unusual participation levels
    - Volatility regime filtering using 30-day vs 90-day volatility ratio to select optimal trading environments
    - Multiplicative combination targets stocks with aligned multi-timeframe trends, extreme volume confirmation,
      and favorable volatility conditions for sustained price movements
    - Designed to capture stocks where momentum signals are reinforced across multiple time horizons
    """
    
    # Multi-timeframe momentum convergence
    momentum_short = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_medium = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    momentum_long = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
    
    # Volume extreme detection using percentile ranking
    volume_rank = df['volume'].rolling(window=30).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
    )
    
    # Volatility regime filtering: medium-term vs long-term volatility ratio
    volatility_30d = ((df['high'] - df['low']) / df['close']).rolling(window=30).mean()
    volatility_90d = ((df['high'] - df['low']) / df['close']).rolling(window=90).mean()
    volatility_regime = volatility_90d / (volatility_30d + 1e-7)
    
    # Combined factor: aligned multi-timeframe momentum amplified by volume extremes and volatility regime
    trend_convergence = momentum_short * momentum_medium * momentum_long
    factor = trend_convergence * volume_rank * volatility_regime
    
    return factor
