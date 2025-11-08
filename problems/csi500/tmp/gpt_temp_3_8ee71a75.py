import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Momentum with Adaptive Volume Weighting alpha factor
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
        Index should be datetime
    
    Returns:
    pandas Series with alpha factor values
    """
    
    # Multi-Timeframe Raw Momentum
    df['M3'] = df['close'] / df['close'].shift(3) - 1
    df['M8'] = df['close'] / df['close'].shift(8) - 1
    df['M21'] = df['close'] / df['close'].shift(21) - 1
    
    # Volatility Normalization
    df['Range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['Vol'] = df['Range'].rolling(window=5).std()
    
    # Volatility-Adjusted Momentum
    df['VM3'] = df['M3'] / (df['Vol'] + 0.0001)
    df['VM8'] = df['M8'] / (df['Vol'] + 0.0001)
    df['VM21'] = df['M21'] / (df['Vol'] + 0.0001)
    
    # Volume Analysis
    # Historical Volume Context (20-day percentile rank)
    def percentile_rank(series):
        if len(series) < 20:
            return np.nan
        current_vol = series.iloc[-1]
        window_vol = series.iloc[:-1]
        return (window_vol <= current_vol).sum() / len(window_vol) * 100
    
    df['VolPct'] = df['volume'].rolling(window=20).apply(percentile_rank, raw=False)
    
    # Volume Trend (5-day)
    df['VolTrend'] = df['volume'] / df['volume'].shift(4) - 1
    
    # Adaptive Volume Weight
    df['Base_Weight'] = (df['VolPct'] / 100) ** (1/3)
    df['Trend_Multiplier'] = 1 + np.sign(df['VolTrend']) * np.abs(df['VolTrend']) ** (1/2)
    df['VolWeight'] = df['Base_Weight'] * df['Trend_Multiplier']
    
    # Momentum Combination
    df['MomentumProduct'] = df['VM3'] * df['VM8'] * df['VM21']
    df['MomentumBlend'] = np.sign(df['MomentumProduct']) * np.abs(df['MomentumProduct']) ** (1/3)
    
    # Final Alpha Factor
    alpha = df['MomentumBlend'] * df['VolWeight']
    
    return alpha
