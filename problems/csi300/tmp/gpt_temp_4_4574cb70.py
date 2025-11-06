import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Divergence
    # Short-term Reversal
    short_term_reversal = ((df['close'] - df['close'].shift(1)) / 
                          (df['high'] - df['low'])) * (df['volume'] / df['volume'].shift(5))
    
    # Medium-term Momentum
    high_low_range_sum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 10:
            high_low_range_sum.iloc[i] = (df['high'].iloc[i-9:i+1] - df['low'].iloc[i-9:i+1]).sum()
    medium_term_momentum = (df['close'] - df['close'].shift(10)) / high_low_range_sum
    
    # Acceleration
    acceleration = ((df['close'] - 2 * df['close'].shift(5) + df['close'].shift(10)) / 
                   (df['high'] - df['low']))
    
    # Combine Price Momentum Divergence
    price_momentum_divergence = short_term_reversal * medium_term_momentum * acceleration
    
    # Volume-Price Divergence
    # Volume Spike
    volume_spike = ((df['volume'] / df['volume'].shift(1)) * 
                   np.abs(df['close'] - df['open']) / (df['high'] - df['low']))
    
    # Price Efficiency
    price_efficiency = (((df['close'] - df['open']) ** 2) / (df['high'] - df['low']) * 
                       df['volume'] / df['amount'])
    
    # Divergence Strength
    divergence_strength = ((df['volume'] / df['volume'].shift(5)) * 
                          (df['close'] - df['close'].shift(1)) / 
                          np.abs(df['close'].shift(1) - df['close'].shift(2)))
    
    # Combine Volume-Price Divergence
    volume_price_divergence = volume_spike * price_efficiency * divergence_strength
    
    # Opening Gap Dynamics
    # Gap Persistence
    gap_persistence = (((df['open'] - df['close'].shift(1)) / 
                       (df['high'].shift(1) - df['low'].shift(1))) * 
                      (df['volume'] / df['volume'].shift(1)))
    
    # Gap Filling
    gap_filling = ((np.abs(df['close'] - df['open']) / np.abs(df['open'] - df['close'].shift(1))) * 
                  ((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))))
    
    # Intraday Recovery
    intraday_recovery = (((df['close'] - df['low']) / (df['high'] - df['low'])) * 
                        np.abs(df['open'] - df['close'].shift(1)))
    
    # Combine Opening Gap Dynamics
    opening_gap_dynamics = gap_persistence * gap_filling * intraday_recovery
    
    # Final Alpha
    alpha = price_momentum_divergence * volume_price_divergence * opening_gap_dynamics
    
    return alpha
