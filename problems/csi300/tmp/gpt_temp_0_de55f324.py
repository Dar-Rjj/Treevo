import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Volume-Confirmed Momentum
    # Short-term Momentum (3-day)
    price_momentum_3d = df['close'] / df['close'].shift(3) - 1
    volume_momentum_3d = df['volume'] / df['volume'].shift(3) - 1
    short_term_signal = price_momentum_3d * volume_momentum_3d
    
    # Medium-term Confirmation (8-day)
    price_trend_8d = df['close'] / df['close'].shift(8) - 1
    volume_trend_8d = df['volume'] / df['volume'].shift(8) - 1
    momentum_factor = short_term_signal * (price_trend_8d * volume_trend_8d)
    
    # Price Efficiency with Volume Validation
    # Efficiency Calculation
    net_price_move = df['close'] - df['close'].shift(5)
    
    # Calculate total range for t-4 to t
    total_range = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            total_range.iloc[i] = (df['high'].iloc[i-4:i+1] - df['low'].iloc[i-4:i+1]).sum()
    
    efficiency_ratio = net_price_move / total_range
    
    # Volume Validation
    volume_ratio_5d = df['volume'] / df['volume'].rolling(window=5, min_periods=5).mean()
    efficiency_factor = efficiency_ratio * volume_ratio_5d
    
    # Directional Flow Consistency
    # Daily Flow Direction
    flow_sign = np.sign(df['close'] - df['close'].shift(1))
    daily_flow = flow_sign * df['amount']
    
    # Multi-day Consistency
    flow_sum_3d = daily_flow.rolling(window=3, min_periods=3).sum()
    
    # Direction Count: count(flow_sign_i = flow_sign_t for i=t-2 to t-1)
    direction_count = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 2:
            current_sign = flow_sign.iloc[i]
            prev_signs = flow_sign.iloc[i-2:i]
            direction_count.iloc[i] = (prev_signs == current_sign).sum()
    
    flow_factor = flow_sum_3d * direction_count
    
    # Breakout Strength with Volume Support
    # Breakout Position
    high_5d = df['high'].rolling(window=5, min_periods=5).max()
    low_5d = df['low'].rolling(window=5, min_periods=5).min()
    breakout_ratio = (df['close'] - low_5d) / (high_5d - low_5d)
    
    # Volume Support
    volume_ratio_breakout = df['volume'] / df['volume'].rolling(window=5, min_periods=5).mean()
    breakout_factor = breakout_ratio * volume_ratio_breakout
    
    # Volatility-Scaled Price-Volume Sync
    # Volatility Context
    tr = np.maximum(df['high'] - df['low'], 
                   np.maximum(abs(df['high'] - df['close'].shift(1)), 
                             abs(df['low'] - df['close'].shift(1))))
    avg_tr_5d = tr.rolling(window=5, min_periods=5).mean()
    volatility_ratio = tr / avg_tr_5d
    
    # Price-Volume Sync
    price_change = df['close'] / df['close'].shift(1) - 1
    volume_change = df['volume'] / df['volume'].shift(1) - 1
    sync_factor = (price_change * volume_change) * volatility_ratio
    
    # Combine all factors with equal weighting
    final_factor = (momentum_factor + efficiency_factor + flow_factor + 
                   breakout_factor + sync_factor) / 5
    
    return final_factor
