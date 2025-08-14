import pandas as pd

def heuristics_v2(df):
    # Calculate the 20-day and 100-day weighted moving averages of the close price
    weights_20 = np.arange(1, 21)
    wma_20 = df['close'].rolling(window=20).apply(lambda prices: np.dot(prices, weights_20)/weights_20.sum(), raw=True)
    
    weights_100 = np.arange(1, 101)
    wma_100 = df['close'].rolling(window=100).apply(lambda prices: np.dot(prices, weights_100)/weights_100.sum(), raw=True)
    
    # Calculate the ratio between the WMAs
    wma_ratio = wma_20 / wma_100
    
    # Calculate the true range
    df['True Range'] = df['high'] - df['low']
    df['True Range'] = df[['True Range', 'high'] - df['close'].shift(1), df['close'].shift(1) - df['low']].max(axis=1)
    
    # Calculate the 30-day average true range
    atr_30 = df['True Range'].rolling(window=30).mean()
    
    # Generate the heuristic matrix by multiplying the WMA ratio with the ATR
    heuristics_matrix = wma_ratio * atr_30
    
    return heuristics_matrix
