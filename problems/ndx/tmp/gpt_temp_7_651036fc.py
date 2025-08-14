import pandas as pd

def heuristics_v2(df):
    heuristics_matrix = pd.Series(index=df.index)
    
    # Example Heuristics Calculation
    # 1. Price Momentum (Close - Open) / Open
    heuristics_matrix['momentum'] = (df['close'] - df['open']) / df['open']
    
    # 2. Volume Trend (Volume today - Volume yesterday) / Volume yesterday
    heuristics_matrix['volume_trend'] = df['volume'].pct_change()
    
    # 3. Volatility (High - Low) / Close
    heuristics_matrix['volatility'] = (df['high'] - df['low']) / df['close']
    
    # 4. Simple Moving Average (SMA) Difference (SMA(5) - SMA(20)) / SMA(20)
    sma_5 = df['close'].rolling(window=5).mean()
    sma_20 = df['close'].rolling(window=20).mean()
    heuristics_matrix['sma_diff'] = (sma_5 - sma_20) / sma_20
    
    # Combine heuristics into a single factor
    weights = [0.4, 0.2, 0.2, 0.2]  # Example weights
    heuristics_matrix = (heuristics_matrix * weights).sum(axis=1)
    
    return heuristics_matrix
