import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price momentum, volume confirmation, and volatility normalization.
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Alpha factor values indexed by date
    """
    
    # Price Momentum Components
    short_momentum = df['close'] / df['close'].shift(5) - 1
    
    medium_momentum = df['close'] / df['close'].shift(10) - 1
    
    # Momentum consistency: count positive momentum days over t-4 to t
    momentum_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        if i >= 4:
            recent_momentum = [df['close'].iloc[i-j] / df['close'].iloc[i-j-1] - 1 for j in range(5)]
            momentum_consistency.iloc[i] = sum(1 for mom in recent_momentum if mom > 0)
    
    # Volume Confirmation Framework
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    
    price_volume_alignment = (np.sign(short_momentum) == np.sign(volume_momentum)).astype(int)
    
    volume_intensity = np.abs(volume_momentum)
    
    # Volatility Context Assessment
    daily_range = (df['high'] - df['low']) / df['close']
    
    # Average range over t-4 to t
    avg_range = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        if i >= 4:
            ranges = [(df['high'].iloc[i-j] - df['low'].iloc[i-j]) / df['close'].iloc[i-j] for j in range(5)]
            avg_range.iloc[i] = np.mean(ranges)
    
    volatility_regime = daily_range / avg_range
    
    # Integrated Factor Construction
    base_momentum = short_momentum * momentum_consistency
    
    # Volume-adjusted momentum
    volume_adjusted = base_momentum * (1 + volume_intensity) * price_volume_alignment + \
                     base_momentum * (1 - price_volume_alignment)
    
    # Volatility-normalized signal
    volatility_normalized = volume_adjusted / volatility_regime
    
    # Handle infinite values and NaN
    volatility_normalized = volatility_normalized.replace([np.inf, -np.inf], np.nan)
    
    return volatility_normalized
