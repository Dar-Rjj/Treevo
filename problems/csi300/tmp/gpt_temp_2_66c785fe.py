import pandas as pd
import pandas as pd

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Novel alpha factor: Multi-timeframe momentum convergence with volume regime detection and volatility-adjusted efficiency
    # Economic rationale: Stocks with aligned momentum across short, medium, and long timeframes, 
    # confirmed by volume regime shifts, and exhibiting volatility-dampened price efficiency tend to have persistent returns
    
    # Multi-timeframe momentum convergence (3, 8, 21 days - Fibonacci sequence for diverse horizons)
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_8d = df['close'] / df['close'].shift(8) - 1
    momentum_21d = df['close'] / df['close'].shift(21) - 1
    
    # Momentum convergence score (product captures alignment across timeframes)
    momentum_convergence = momentum_3d * momentum_8d * momentum_21d
    
    # Volume regime detection using rolling percentiles (more robust than fixed thresholds)
    volume_20d = df['volume'].rolling(20)
    volume_percentile = volume_20d.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Volatility-adjusted price efficiency
    # True range for volatility normalization
    true_range = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    # Efficiency ratio: price movement relative to volatility
    net_price_change = abs(df['close'] - df['close'].shift(1))
    efficiency_ratio = net_price_change / (true_range + 1e-7)
    
    # Direction-aware efficiency (positive for uptrends, negative for downtrends)
    direction = (df['close'] > df['close'].shift(1)).astype(int) * 2 - 1
    directional_efficiency = direction * efficiency_ratio
    
    # Combine: momentum convergence amplified by volume regime and directional efficiency
    # Volume percentile provides regime confirmation, directional efficiency adds quality filter
    factor = momentum_convergence * volume_percentile * directional_efficiency
    
    return factor
