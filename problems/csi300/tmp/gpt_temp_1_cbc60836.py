import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volume-Confirmed Momentum Persistence factor
    Combines price momentum, momentum persistence, and volume confirmation
    with volatility adjustment
    """
    df = data.copy()
    
    # 5-Day Price Momentum
    df['momentum_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Momentum Persistence - count consecutive days with same momentum direction
    df['momentum_direction'] = np.sign(df['momentum_5d'])
    df['momentum_persistence'] = 0
    
    current_direction = None
    current_count = 0
    
    for i in range(len(df)):
        if pd.isna(df['momentum_direction'].iloc[i]):
            df.loc[df.index[i], 'momentum_persistence'] = 0
            current_count = 0
        elif df['momentum_direction'].iloc[i] == current_direction:
            current_count += 1
            df.loc[df.index[i], 'momentum_persistence'] = current_count
        else:
            current_direction = df['momentum_direction'].iloc[i]
            current_count = 1
            df.loc[df.index[i], 'momentum_persistence'] = current_count
    
    # Volume Trend (3-day)
    df['volume_trend'] = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    
    # Volume-Price Alignment Score
    df['volume_price_alignment'] = 1.0  # default neutral
    
    # Same direction: both positive or both negative
    same_direction_mask = (df['momentum_5d'] > 0) & (df['volume_trend'] > 0) | \
                         (df['momentum_5d'] < 0) & (df['volume_trend'] < 0)
    
    # Opposite direction
    opposite_direction_mask = (df['momentum_5d'] > 0) & (df['volume_trend'] < 0) | \
                             (df['momentum_5d'] < 0) & (df['volume_trend'] > 0)
    
    df.loc[same_direction_mask, 'volume_price_alignment'] = 2.0
    df.loc[opposite_direction_mask, 'volume_price_alignment'] = 0.5
    
    # Core Signal: Volume-Confirmed Momentum
    df['core_signal'] = df['momentum_5d'] * df['volume_price_alignment'] * df['momentum_persistence']
    
    # Risk Adjustment: Daily Range Normalized by Close
    df['daily_range_normalized'] = (df['high'] - df['low']) / df['close']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['core_signal'] / df['daily_range_normalized']
    
    # Replace inf and NaN values
    df['alpha_factor'] = df['alpha_factor'].replace([np.inf, -np.inf], np.nan)
    df['alpha_factor'] = df['alpha_factor'].fillna(0)
    
    return df['alpha_factor']
