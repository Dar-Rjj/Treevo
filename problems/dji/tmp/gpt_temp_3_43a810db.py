import pandas as pd
    
    # Momentum indicator: 10-day simple moving average
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    
    # Volatility indicator: 10-day standard deviation of daily returns
    df['Volatility_10'] = df['close'].pct_change().rolling(window=10).std()
    
    # Volume indicator: 5-day average volume
    df['Avg_Vol_5'] = df['volume'].rolling(window=5).mean()
    
    # Constructing the heuristics matrix
    heuristics_matrix = pd.concat([df['SMA_10'], df['Volatility_10'], df['Avg_Vol_5']], axis=1)
    heuristics_matrix.columns = ['Momentum', 'Volatility', 'Volume']
    
    return heuristics_matrix
```
Note: The function `heuristics_v2` is designed to illustrate the process, but it return heuristics_matrix
