import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate price and volume features
    df = df.copy()
    
    # Price features
    df['price_change'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_acceleration'] = (df['volume_ma_5'] - df['volume_ma_10']) / df['volume_ma_10']
    
    # Bullish divergence detection
    # Price bottom formation (local minima in close price)
    df['price_minima'] = (df['close'] < df['close'].shift(1)) & (df['close'] < df['close'].shift(-1))
    df['price_bottom_strength'] = df['close'].rolling(window=5).apply(
        lambda x: (x.iloc[2] - min(x.iloc[0], x.iloc[4])) / x.iloc[2] if len(x) == 5 else 0
    )
    
    # Rising volume during price consolidation
    df['consolidation'] = df['high_low_range'].rolling(window=3).mean() < df['high_low_range'].rolling(window=10).mean() * 0.8
    df['volume_rising'] = (df['volume'] > df['volume'].shift(1)) & (df['volume_ma_5'] > df['volume_ma_10'])
    
    bullish_divergence = (
        (df['price_minima'] | (df['price_bottom_strength'] > 0.02)) & 
        df['consolidation'] & 
        df['volume_rising']
    )
    df['bullish_strength'] = np.where(
        bullish_divergence,
        df['price_bottom_strength'] * df['volume_acceleration'].clip(lower=0),
        0
    )
    
    # Bearish divergence detection
    # Price top formation (local maxima in close price)
    df['price_maxima'] = (df['close'] > df['close'].shift(1)) & (df['close'] > df['close'].shift(-1))
    df['price_top_strength'] = df['close'].rolling(window=5).apply(
        lambda x: (max(x.iloc[0], x.iloc[4]) - x.iloc[2]) / x.iloc[2] if len(x) == 5 else 0
    )
    
    # Declining volume during price advances
    df['price_advancing'] = df['close'] > df['close'].rolling(window=5).mean()
    df['volume_declining'] = (df['volume'] < df['volume'].shift(1)) & (df['volume_ma_5'] < df['volume_ma_10'])
    
    bearish_divergence = (
        (df['price_maxima'] | (df['price_top_strength'] > 0.02)) & 
        df['price_advancing'] & 
        df['volume_declining']
    )
    df['bearish_strength'] = np.where(
        bearish_divergence,
        df['price_top_strength'] * abs(df['volume_acceleration'].clip(upper=0)),
        0
    )
    
    # Signal integration with time decay
    decay_rate = 0.9
    df['bullish_momentum'] = 0.0
    df['bearish_momentum'] = 0.0
    
    for i in range(1, len(df)):
        df.loc[df.index[i], 'bullish_momentum'] = (
            decay_rate * df.loc[df.index[i-1], 'bullish_momentum'] + 
            df.loc[df.index[i], 'bullish_strength']
        )
        df.loc[df.index[i], 'bearish_momentum'] = (
            decay_rate * df.loc[df.index[i-1], 'bearish_momentum'] + 
            df.loc[df.index[i], 'bearish_strength']
        )
    
    # Final factor: bullish minus bearish momentum
    factor = df['bullish_momentum'] - df['bearish_momentum']
    
    return factor
