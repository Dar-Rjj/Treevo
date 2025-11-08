import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Multi-Timeframe Momentum with Volume Confirmation
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns for different timeframes
    df['ret_3'] = df['close'] / df['close'].shift(3) - 1
    df['ret_8'] = df['close'] / df['close'].shift(8) - 1
    df['ret_21'] = df['close'] / df['close'].shift(21) - 1
    
    # Calculate True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate volatility (5-day rolling std of TR/Close[t-1])
    df['tr_ratio'] = df['tr'] / df['close'].shift(1)
    df['volatility'] = df['tr_ratio'].rolling(window=5).std()
    
    # Volatility-adjusted returns
    df['m3_adj'] = df['ret_3'] / (df['volatility'] + 0.001)
    df['m8_adj'] = df['ret_8'] / (df['volatility'] + 0.001)
    df['m21_adj'] = df['ret_21'] / (df['volatility'] + 0.001)
    
    # Volume percentile analysis (20-day window)
    df['volume_percentile'] = df['volume'].rolling(window=20).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    df['vp'] = (df['volume_percentile'] / 100) ** (1/3)
    
    # Volume trend confirmation
    df['volume_change'] = df['volume'] / df['volume'].shift(5) - 1
    df['vm'] = np.sign(df['volume_change']) * np.minimum(1, abs(df['volume_change']))
    
    # Multiplicative momentum blend
    df['momentum_product'] = df['m3_adj'] * df['m8_adj'] * df['m21_adj']
    df['momentum_blend'] = np.sign(df['momentum_product']) * np.minimum(
        10, abs(df['momentum_product']) ** (1/3)
    )
    
    # Volume confidence
    df['volume_confidence'] = df['vp'] * (1 + 0.5 * df['vm'])
    
    # Final alpha signal
    df['alpha'] = df['momentum_blend'] * df['volume_confidence']
    
    # Fill result series
    result = df['alpha'].copy()
    
    return result
