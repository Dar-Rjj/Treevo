import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Adaptive multi-timeframe momentum with volume-to-median confirmation
    and volatility dampening using rolling percentiles.
    
    Economic rationale: Combines short-to-medium term momentum across multiple timeframes
    with volume confirmation relative to rolling median, while using volatility percentiles
    for robust signal scaling in different market regimes.
    """
    
    # Adaptive window sizing based on data availability
    data_length = len(df)
    short_window = min(5, max(3, data_length // 10))
    medium_window = min(10, max(5, data_length // 5))
    vol_window = min(20, max(10, data_length // 4))
    
    # Multi-timeframe momentum with adaptive windows
    momentum_short = df['close'] / df['close'].shift(short_window) - 1
    momentum_medium = df['close'] / df['close'].shift(medium_window) - 1
    
    # Volume-to-median ratio using adaptive rolling window
    volume_median = df['volume'].rolling(window=vol_window, min_periods=5).median()
    volume_ratio = df['volume'] / (volume_median + 1e-7)
    
    # Rolling volatility using high-low range percentiles
    daily_range_pct = (df['high'] - df['low']) / df['close']
    vol_percentile = daily_range_pct.rolling(window=vol_window, min_periods=5).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.5)).astype(float) if len(x.dropna()) >= 5 else np.nan
    )
    
    # Amount-based confirmation (trading activity intensity)
    amount_median = df['amount'].rolling(window=vol_window, min_periods=5).median()
    amount_ratio = df['amount'] / (amount_median + 1e-7)
    
    # Combined momentum with volume and amount confirmation
    combined_momentum = (0.5 * momentum_short + 0.5 * momentum_medium) * volume_ratio * amount_ratio
    
    # Final factor: momentum signals dampened by volatility regime
    # Lower volatility percentiles amplify positive momentum signals
    factor = combined_momentum / (vol_percentile + 1e-7)
    
    return factor
