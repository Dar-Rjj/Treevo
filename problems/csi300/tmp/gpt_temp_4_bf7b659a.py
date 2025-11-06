import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate Momentum Acceleration
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    df['momentum_acceleration'] = df['return_5d'] - df['return_20d']
    
    # Assess Liquidity Conditions
    df['daily_range'] = df['high'] - df['low']
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['range_efficiency_ratio'] = df['daily_range'] / df['true_range']
    
    # Generate Final Signal
    df['volume_5d'] = df['volume'].rolling(window=5).mean()
    df['volume_20d'] = df['volume'].rolling(window=20).mean()
    df['volume_surge_ratio'] = df['volume_5d'] / df['volume_20d']
    
    # Normalize components for combination
    df['acceleration_rank'] = df['momentum_acceleration'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['efficiency_rank'] = df['range_efficiency_ratio'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['volume_rank'] = df['volume_surge_ratio'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Combine signals with equal weighting
    df['factor'] = df['acceleration_rank'] * df['efficiency_rank'] * df['volume_rank']
    
    return df['factor']
