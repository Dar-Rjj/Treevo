import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum acceleration with volume efficiency and volatility normalization
    # Uses non-linear transforms and relative comparisons for better predictive power
    
    # Price momentum acceleration (2nd derivative of price)
    short_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    long_momentum = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    momentum_acceleration = short_momentum - long_momentum
    
    # Volume efficiency (volume relative to price movement)
    price_range_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    volume_relative = df['volume'] / df['volume'].rolling(window=15).median()
    volume_efficiency = price_range_efficiency * volume_relative
    
    # Volatility normalization using rolling percentiles
    recent_volatility = (df['high'] - df['low']).rolling(window=10).std()
    vol_percentile = recent_volatility.rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(float)
    )
    
    # Non-linear combination with economic rationale:
    # Strong momentum acceleration with efficient volume confirmation,
    # penalized during high volatility regimes
    factor = (momentum_acceleration * np.tanh(volume_efficiency)) / (1 + vol_percentile)
    
    return factor
