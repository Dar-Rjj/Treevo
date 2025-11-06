import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Price Reversal Strength
    # Short-Term Reversal (t-1 vs t-5 return)
    ret_1 = df['close'].pct_change(1)
    ret_5 = df['close'].pct_change(5)
    short_term_reversal = ret_1 - ret_5
    
    # Medium-Term Reversal (t-1 vs t-20 return)
    ret_20 = df['close'].pct_change(20)
    medium_term_reversal = ret_1 - ret_20
    
    # Combine Reversal Signals (weighted average)
    reversal_strength = 0.6 * short_term_reversal + 0.4 * medium_term_reversal
    
    # Analyze Liquidity Acceleration
    # Volume Velocity (t vs t-5 volume change)
    volume_velocity = df['volume'] / df['volume'].shift(5) - 1
    
    # Amount Momentum (t vs t-10 amount change)
    amount_momentum = df['amount'] / df['amount'].shift(10) - 1
    
    # Calculate Liquidity Composite
    liquidity_composite = volume_velocity * amount_momentum
    
    # Measure Volatility Regime
    # Calculate True Range (t)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Compute Volatility Ratio (current TR vs 10-day median TR)
    tr_median = true_range.rolling(window=10, min_periods=5).median()
    volatility_ratio = true_range / tr_median
    
    # Classify Regime (high/low volatility threshold)
    volatility_regime = np.where(volatility_ratio > 1.2, volatility_ratio, 1.0)
    
    # Integrate Factors
    # Weight Reversal by Liquidity Acceleration
    weighted_reversal = reversal_strength * liquidity_composite
    
    # Adjust for Volatility Regime
    volatility_adjusted = weighted_reversal / volatility_regime
    
    # Apply Dynamic Scaling (rolling percentile rank)
    factor = volatility_adjusted.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    return factor
