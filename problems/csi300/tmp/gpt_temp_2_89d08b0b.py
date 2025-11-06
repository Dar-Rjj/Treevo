import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate short-term price momentum
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Calculate volume percentile rank over past 20 days
    df['volume_rank'] = df['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Compute volume-weighted momentum
    df['vw_momentum_5d'] = df['momentum_5d'] * df['volume_rank']
    df['vw_momentum_10d'] = df['momentum_10d'] * df['volume_rank']
    
    # Detect asymmetric momentum patterns
    bullish_condition = (df['vw_momentum_5d'] > df['vw_momentum_10d']) & (df['close'] > df['open'])
    bearish_condition = (df['vw_momentum_5d'] < df['vw_momentum_10d']) & (df['close'] < df['open'])
    
    # Calculate momentum divergence
    momentum_divergence = df['vw_momentum_5d'] - df['vw_momentum_10d']
    
    # Combine bullish and bearish signals
    factor = np.where(
        bullish_condition,
        momentum_divergence * df['volume_rank'],
        np.where(
            bearish_condition,
            momentum_divergence * df['volume_rank'],
            0
        )
    )
    
    return pd.Series(factor, index=df.index)
