importance.}

```python
import pandas as pd

def heuristics_v2(df):
    # Calculate simple returns for momentum with different time windows
    short_returns = df['close'].pct_change(5)
    medium_returns = df['close'].pct_change(20)
    long_returns = df['close'].pct_change(60)
    
    # Average returns with weights
    avg_returns = 0.3 * short_returns + 0.4 * medium_returns + 0.3 * long_returns
    
    # Calculate relative volatility (standard deviation of returns over a longer period)
    relative_volatility = avg_returns.rolling(window=120).std()
    
    # Calculate liquidity (volume divided by close price)
    liquidity = df['volume'] / df['close']
    
    # Combine factors into a heuristics matrix using a weighted sum
    heuristics_matrix = 0.5 * avg_returns + 0.3 * relative_volatility + 0.2 * liquidity
    
    return heuristics_matrix
