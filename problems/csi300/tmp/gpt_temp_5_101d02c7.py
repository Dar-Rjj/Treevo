import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Regime-Adaptive Momentum with Volume-Price Divergence Detection
    
    Enhances volatility-normalized momentum by incorporating regime detection through 
    rolling volatility percentiles, then identifies volume-price divergences that may 
    signal momentum sustainability or reversal. This captures stocks with meaningful 
    price movements relative to their volatility regime, while detecting potential 
    momentum exhaustion through volume patterns.
    
    Factor interpretation: Identifies stocks experiencing regime-relative price moves 
    with additional insights from volume-price relationship dynamics, providing signals 
    about both momentum strength and potential sustainability.
    """
    # 10-day price momentum
    momentum = df['close'] - df['close'].shift(10)
    
    # Rolling volatility using 15-day standard deviation of returns
    returns = df['close'].pct_change()
    volatility = returns.rolling(15).std()
    
    # Volatility regime detection using rolling percentile (20-day window)
    volatility_regime = volatility.rolling(20).apply(lambda x: (x[-1] - x.mean()) / (x.std() + 1e-7))
    
    # Volume-price divergence: compare volume trend with price trend
    volume_trend = df['volume'].rolling(10).apply(lambda x: (x[-1] - x[0]) / (x.mean() + 1e-7))
    price_trend = df['close'].rolling(10).apply(lambda x: (x[-1] - x[0]) / (x.mean() + 1e-7))
    volume_price_divergence = volume_trend - price_trend
    
    # Intraday momentum confirmation: normalized close position with range expansion
    intraday_range = df['high'] - df['low']
    range_expansion = intraday_range / intraday_range.rolling(10).mean()
    close_position = (df['close'] - df['low']) / (intraday_range + 1e-7)
    intraday_confirmation = close_position * range_expansion
    
    # Combined factor: regime-adaptive momentum with divergence detection
    alpha_factor = (momentum / (volatility + 1e-7)) * volatility_regime * volume_price_divergence * intraday_confirmation
    
    return alpha_factor
