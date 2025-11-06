import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum alignment with volume-pressure weighted volatility adaptation
    # Economic intuition: Aligned momentum signals across timeframes, confirmed by volume-pressure,
    # and weighted by volatility regime provide robust predictive signals
    
    # Short-term momentum (5-day)
    mom_short = df['close'] / df['close'].shift(5) - 1
    
    # Medium-term momentum (10-day)  
    mom_medium = df['close'] / df['close'].shift(10) - 1
    
    # Long-term momentum (20-day)
    mom_long = df['close'] / df['close'].shift(20) - 1
    
    # Calculate momentum alignment (product of z-score aligned momentums)
    # Smooth transformation using tanh to bound signals
    mom_alignment = (
        np.tanh(mom_short * 10) * 
        np.tanh(mom_medium * 8) * 
        np.tanh(mom_long * 6)
    )
    
    # Volume-pressure confirmation (amount-weighted volume momentum)
    volume_pressure = (df['amount'] / df['volume']) * (df['volume'] / df['volume'].shift(5) - 1)
    # Smooth volume pressure with bounded transformation
    volume_confirmation = np.tanh(volume_pressure * 5)
    
    # Volatility-adaptive weighting using range efficiency
    daily_range = df['high'] - df['low']
    range_efficiency = (df['close'] - df['open']) / (daily_range + 1e-7)
    # Volatility regime detection using rolling percentiles
    vol_regime = daily_range.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-7))
    # Adaptive weight: higher weight for efficient moves in moderate volatility
    volatility_weight = np.exp(-2 * abs(vol_regime - 0.5)) * (1 + abs(range_efficiency))
    
    # Combine factors with economic rationale:
    # Aligned momentum amplified by volume pressure, weighted by volatility regime efficiency
    factor = mom_alignment * volume_confirmation * volatility_weight
    
    return factor
