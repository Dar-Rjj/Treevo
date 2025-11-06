import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 5-day rolling volatility (price range normalized by close)
    volatility = (df['high'] - df['low']).rolling(window=5, min_periods=3).mean() / df['close']
    
    # Volatility regime classification using rolling percentiles
    vol_regime = volatility.rolling(window=20, min_periods=10).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else (-1 if x.iloc[-1] < x.quantile(0.3) else 0)
    )
    
    # Short-term momentum (3-day price return)
    momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Mean reversion component (current price vs 5-day moving average)
    ma_5 = df['close'].rolling(window=5, min_periods=3).mean()
    mean_reversion = (ma_5 - df['close']) / ma_5
    
    # Price-volume divergence (price change vs volume change correlation)
    price_change = df['close'].pct_change(periods=3)
    volume_change = df['volume'].pct_change(periods=3)
    pv_divergence = price_change * volume_change
    
    # Liquidity factor (amount per volume unit)
    liquidity = df['amount'] / (df['volume'] + 1e-7)
    liquidity_momentum = (liquidity - liquidity.shift(3)) / (liquidity.shift(3) + 1e-7)
    
    # Regime-aware factor combination
    factor = vol_regime * (
        # High volatility: emphasize mean reversion and liquidity
        (vol_regime == 1) * (mean_reversion * liquidity_momentum) +
        # Low volatility: emphasize momentum and price-volume confirmation  
        (vol_regime == -1) * (momentum * pv_divergence) +
        # Normal volatility: balanced approach
        (vol_regime == 0) * (momentum * mean_reversion * pv_divergence)
    )
    
    return factor
