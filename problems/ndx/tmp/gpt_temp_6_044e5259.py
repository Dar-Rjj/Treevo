import pandas as pd
    import numpy as np
    
    def calculate_volatility(returns, window=5):
        return returns.rolling(window).std()
    
    price_changes = df['close'].pct_change().fillna(0)
    volatility = calculate_volatility(price_changes)
    volume_weighted = (df['volume'] * price_changes.shift(1)).rolling(window=10).mean()
    heuristics_matrix = (volume_weighted / (volatility + 1e-6)).fillna(0)
    
    return heuristics_matrix
