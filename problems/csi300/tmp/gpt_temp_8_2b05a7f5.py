import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Order flow imbalance
    typical_price = (high + low + close) / 3
    dollar_volume = typical_price * volume
    ofi = (close.diff(3) * dollar_volume.diff(3)).rolling(8).sum()
    
    # Price impact efficiency
    price_range = high - low
    efficiency_ratio = close.diff(5).abs() / price_range.rolling(5).sum()
    impact_efficiency = ofi / (price_range * volume + 1e-8) * efficiency_ratio
    
    # Volume-accelerated trend persistence
    volume_trend = volume.rolling(15).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    price_trend = close.rolling(15).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    trend_persistence = np.sign(price_trend) * volume_trend * close.diff(3)
    
    # Liquidity-constrained combination
    liquidity_filter = dollar_volume.rolling(10).mean()
    constrained_momentum = impact_efficiency * trend_persistence * np.log(liquidity_filter + 1e-8)
    
    heuristics_matrix = constrained_momentum
    
    return heuristics_matrix
