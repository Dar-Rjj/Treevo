import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Scaled Momentum Acceleration
    # Momentum Component
    close = df['close']
    current_momentum = close / close.shift(5) - 1
    previous_momentum = close.shift(5) / close.shift(10) - 1
    momentum_acceleration = current_momentum - previous_momentum
    
    # Volume Scaling Component
    volume = df['volume']
    volume_intensity = volume / volume.shift(20) - 1
    
    # Volume Persistence
    volume_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        consecutive_count = 0
        for j in range(i-9, i+1):
            if volume.iloc[j] > volume.iloc[j-1]:
                consecutive_count += 1
            else:
                break
        volume_persistence.iloc[i] = consecutive_count
    
    # Combined Factor 1
    factor1 = momentum_acceleration * volume_intensity * volume_persistence
    
    # Efficiency-Adjusted Price Range
    # Price Efficiency Component
    high = df['high']
    low = df['low']
    close_prev = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    open_price = df['open']
    closing_efficiency = abs(close - open_price) / true_range
    
    # Volume Confirmation
    volume_avg_5 = volume.rolling(window=5).mean()
    volume_confirmation = volume / volume_avg_5
    
    # Efficiency Factor
    factor2 = closing_efficiency * volume_confirmation
    
    # Gap-Filled Momentum Persistence
    # Gap Analysis
    overnight_gap = (open_price - close_prev) / close_prev
    
    gap_filling = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i == 0:
            gap_filling.iloc[i] = 0
            continue
        current_open = open_price.iloc[i]
        prev_close = close_prev.iloc[i]
        current_close = close.iloc[i]
        
        if (current_close >= min(current_open, prev_close) and 
            current_close <= max(current_open, prev_close)):
            gap_filling.iloc[i] = 1
        else:
            gap_filling.iloc[i] = 0
    
    # Momentum Integration
    momentum_3d = close / close.shift(3) - 1
    
    momentum_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        same_sign_count = 0
        for j in range(i-4, i+1):
            if j > 0 and momentum_3d.iloc[j] * momentum_3d.iloc[j-1] > 0:
                same_sign_count += 1
        momentum_consistency.iloc[i] = same_sign_count
    
    # Combined Factor 3
    factor3 = gap_filling * momentum_3d * momentum_consistency
    
    # Volume-Weighted Range Breakout
    # Range Analysis
    daily_range = (high - low) / close
    
    range_breakout = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        prev_high_max = high.iloc[i-4:i].max()
        prev_low_min = low.iloc[i-4:i].min()
        current_close = close.iloc[i]
        
        if current_close > prev_high_max or current_close < prev_low_min:
            range_breakout.iloc[i] = 1
        else:
            range_breakout.iloc[i] = 0
    
    # Volume Validation
    prev_volume_avg = volume.rolling(window=4).mean().shift(1)
    breakout_volume = volume / prev_volume_avg
    
    volume_spike = (volume > 2 * volume.shift(1)).astype(float)
    
    # Breakout Factor
    factor4 = range_breakout * breakout_volume * volume_spike
    
    # Accumulation-Distribution Momentum
    # Money Flow Component
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    # Momentum Integration
    price_change = close - close.shift(1)
    avg_volume = volume.rolling(window=20).mean()
    volume_weighted_momentum = price_change * raw_money_flow / avg_volume
    
    # Accumulation Factor
    factor5 = volume_weighted_momentum.rolling(window=5).sum()
    
    # Combine all factors with equal weights
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
                      factor4.fillna(0) + factor5.fillna(0)) / 5
    
    return combined_factor
