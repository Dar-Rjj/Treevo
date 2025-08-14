import pandas as pd

def heuristics_v2(df):
    # Calculate simple moving averages for different periods
    sma_5 = df['close'].rolling(window=5).mean()
    sma_20 = df['close'].rolling(window=20).mean()
    # Momentum factor: difference between short-term and long-term SMA
    momentum_factor = sma_5 - sma_20
    
    # Volatility factor: standard deviation over a window
    volatility_factor = df['close'].pct_change().rolling(window=20).std()
    
    # Volume trend factor: 1 if today's volume is higher than the 20-day average, else 0
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    volume_trend_factor = (df['volume'] > avg_volume_20).astype(int)
    
    # Combine factors into a single matrix
    heuristics_matrix = pd.DataFrame({'momentum': momentum_factor, 'volatility': volatility_factor, 'volume_trend': volume_trend_factor})
    
    return heuristics_matrix
