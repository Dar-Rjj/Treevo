import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Frequency Fractal Reversal
    # High-frequency
    high_freq = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * 
                (data['volume'] / data['volume'].shift(1).replace(0, 1e-8)) * 
                np.sign(data['close'] - data['open']) * -1)
    
    # Medium-frequency
    close_diff_5 = data['close'] - data['close'].shift(5)
    sum_abs_diff_5 = (abs(data['close'] - data['close'].shift(1)) + 
                     abs(data['close'].shift(1) - data['close'].shift(2)) + 
                     abs(data['close'].shift(2) - data['close'].shift(3)) + 
                     abs(data['close'].shift(3) - data['close'].shift(4)) + 
                     abs(data['close'].shift(4) - data['close'].shift(5)))
    medium_freq = (close_diff_5 / (sum_abs_diff_5 + 1e-8) * 
                  np.sign(data['close'] - data['open']) * -1)
    
    # Low-frequency
    close_diff_20 = data['close'] - data['close'].shift(20)
    sum_abs_diff_20 = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 20:
            sum_abs_diff_20.iloc[i] = sum(abs(data['close'].iloc[j] - data['close'].iloc[j-1]) 
                                        for j in range(i-19, i+1))
    low_freq = (close_diff_20 / (sum_abs_diff_20 + 1e-8) * 
               (data['volume'] / (data['amount'] + 1e-8)) * 
               np.sign(data['close'] - data['open']))
    
    # Bid-Ask Fractal Reversal
    # Bid
    bid = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8) * 
          (data['volume'] / (data['amount'] + 1e-8)) * 
          np.sign(data['close'] - data['open']) * -1)
    
    # Ask
    ask = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8) * 
          (data['volume'] / (data['amount'] + 1e-8)) * 
          np.sign(data['close'] - data['open']) * -1)
    
    # Spread
    hl_diff = data['high'] - data['low']
    hl_diff_prev = hl_diff.shift(1)
    spread = (abs(hl_diff - hl_diff_prev) / (hl_diff_prev + 1e-8) * 
             (data['volume'] / data['volume'].shift(1).replace(0, 1e-8)) * 
             np.sign(data['close'] - data['open']) * -1)
    
    # Volume-Fractal Integration
    # Core
    core = (abs(data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8) * 
           (data['volume'] / (data['amount'] + 1e-8)) * 
           np.sign(data['close'] - data['open']) * -1)
    
    # Momentum
    momentum = ((data['close'] - data['close'].shift(1)) * 
               (data['volume'] / (data['amount'] + 1e-8)) * 
               np.sign(data['close'] - data['open']) * -1)
    
    # Breakout
    breakout = (core * (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * 
               (data['volume'] / (data['amount'] + 1e-8)) * 
               np.sign(data['close'] - data['open']) * -1)
    
    # Price-Volume Divergence
    # Acceleration
    close_diff_1 = data['close'] - data['close'].shift(1)
    close_diff_2 = data['close'].shift(1) - data['close'].shift(2)
    close_diff_3 = data['close'].shift(2) - data['close'].shift(3)
    
    accel_ratio1 = close_diff_1 / (close_diff_2 + 1e-8)
    accel_ratio2 = close_diff_2 / (close_diff_3 + 1e-8)
    acceleration = (accel_ratio1 - accel_ratio2) * np.sign(data['close'] - data['open'])
    
    # Divergence
    price_change_pct = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    volume_change_pct = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 1e-8)
    divergence = (price_change_pct - volume_change_pct) * abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Hierarchical Reversal Classification
    strong_cond = ((high_freq > 0.6) & (medium_freq > 0.5) & (low_freq < -0.3) & 
                  (bid > 0.5) & (ask > 0.5) & (spread < 0.2) & 
                  (momentum > 0) & (core > 0.1))
    
    moderate_cond = (((high_freq > 0.6) | (medium_freq > 0.5) | (low_freq < -0.3) | 
                     (bid > 0.5) | (ask > 0.5) | (spread < 0.2)) & 
                    (momentum > 0) & (core > 0.1))
    
    weak_cond = (momentum > 0) | (core > 0)
    weak_continuation_cond = (momentum < 0) | (core < 0)
    moderate_continuation_cond = (bid < 0.5) & (ask < 0.5)
    strong_continuation_cond = ((high_freq > 0.6) & (medium_freq > 0.5) & (low_freq < -0.3) & 
                               (bid > 0.5) & (ask > 0.5) & (spread < 0.2) & 
                               (momentum < 0))
    
    # Assign scores based on classification
    factor = pd.Series(index=data.index, dtype=float)
    factor[strong_cond] = 3.0
    factor[moderate_cond & ~strong_cond] = 2.0
    factor[weak_cond & ~moderate_cond & ~strong_cond] = 1.0
    factor[weak_continuation_cond & ~weak_cond & ~moderate_cond & ~strong_cond] = -1.0
    factor[moderate_continuation_cond & ~weak_continuation_cond & ~weak_cond & ~moderate_cond & ~strong_cond] = -2.0
    factor[strong_continuation_cond & ~moderate_continuation_cond & ~weak_continuation_cond & ~weak_cond & ~moderate_cond & ~strong_cond] = -3.0
    factor.fillna(0, inplace=True)  # Neutral signals
    
    return factor
