import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum hierarchy
    momentum_5d = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-7)
    momentum_10d = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-7)
    momentum_20d = (df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + 1e-7)
    
    # Momentum acceleration and hierarchy
    momentum_acceleration = momentum_5d - momentum_10d
    momentum_hierarchy = momentum_5d / (momentum_20d + 1e-7)
    
    # Adaptive volatility scaling using rolling percentiles
    volatility_20d = (df['high'] - df['low']).rolling(window=20).std()
    vol_percentile = volatility_20d.rolling(window=60).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 1.0, raw=False)
    adaptive_vol_scaling = 1.0 / (1.0 + vol_percentile * volatility_20d)
    
    # Volume-price trend confirmation
    volume_trend_10d = df['volume'].rolling(window=10).apply(lambda x: np.polyfit(range(10), x, 1)[0], raw=True)
    price_trend_10d = df['close'].rolling(window=10).apply(lambda x: np.polyfit(range(10), x, 1)[0], raw=True)
    volume_price_confirmation = np.tanh(volume_trend_10d * price_trend_10d)
    
    # Range breakout efficiency
    current_range = df['high'] - df['low']
    range_breakout = (df['close'] - df['open']) / (current_range + 1e-7)
    range_efficiency_15d = range_breakout.rolling(window=15).mean()
    range_breakout_signal = range_breakout - range_efficiency_15d
    
    # Liquidity-adjusted amount flow
    amount_momentum_8d = np.log(df['amount'] / (df['amount'].shift(8) + 1e-7))
    volume_adjusted_amount = amount_momentum_8d / (np.log(df['volume'] / (df['volume'].shift(8) + 1e-7)) + 1e-7)
    
    # Multiplicative combination with bounded components
    alpha_factor = (
        momentum_acceleration * 
        momentum_hierarchy * 
        adaptive_vol_scaling * 
        volume_price_confirmation * 
        np.tanh(range_breakout_signal) * 
        np.tanh(volume_adjusted_amount)
    )
    
    return alpha_factor
