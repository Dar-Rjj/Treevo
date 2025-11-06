import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical signals:
    - Momentum-Volume Divergence
    - Volatility Efficiency  
    - Breakout Strength
    - Gap Momentum
    - Price-Volume Acceleration
    - Amount-Price Confirmation
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(2, len(df)):
        # Momentum-Volume Divergence
        price_change_2d = df['close'].iloc[i] - df['close'].iloc[i-2]
        volume_change_2d = df['volume'].iloc[i] - df['volume'].iloc[i-2]
        momentum_volume = price_change_2d * volume_change_2d
        
        # Volatility Efficiency
        price_range_3d = sum(df['high'].iloc[i-j] - df['low'].iloc[i-j] for j in range(3))
        net_movement_3d = sum(df['close'].iloc[i-j] - df['close'].iloc[i-j-1] for j in range(3))
        efficiency_ratio = net_movement_3d / price_range_3d if price_range_3d != 0 else 0
        
        # Breakout Strength
        breakout_distance = df['close'].iloc[i] - max(df['high'].iloc[i-1], df['high'].iloc[i-2])
        volume_surge = df['volume'].iloc[i] - (df['volume'].iloc[i-1] + df['volume'].iloc[i-2]) / 2
        breakout_signal = breakout_distance * volume_surge
        
        # Gap Momentum
        gap_size = df['open'].iloc[i] - df['close'].iloc[i-1]
        intraday_movement = df['close'].iloc[i] - df['open'].iloc[i]
        gap_factor = gap_size * intraday_movement
        
        # Price-Volume Acceleration
        price_change_1d = df['close'].iloc[i] - df['close'].iloc[i-1]
        volume_change_1d = df['volume'].iloc[i] - df['volume'].iloc[i-1]
        acceleration_signal = price_change_1d * volume_change_1d
        
        # Amount-Price Confirmation
        amount_change = df['amount'].iloc[i] - df['amount'].iloc[i-1]
        confirmation_signal = price_change_1d * amount_change
        
        # Combine all signals with equal weights
        combined_signal = (momentum_volume + efficiency_ratio + breakout_signal + 
                          gap_factor + acceleration_signal + confirmation_signal) / 6
        
        result.iloc[i] = combined_signal
    
    return result
