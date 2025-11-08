import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    df = df.copy()
    df['returns_5d'] = df['close'].pct_change(5)
    df['returns_10d'] = df['close'].pct_change(10)
    
    # Volatility clustering strength - 5-day volatility autocorrelation (lag 1)
    df['volatility_5d'] = df['returns_5d'].rolling(window=5).std()
    df['volatility_autocorr'] = df['volatility_5d'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    
    # Momentum decay rate = 5-day ROC / 10-day ROC
    df['momentum_decay'] = df['returns_5d'] / (df['returns_10d'] + 1e-8)
    
    # Price efficiency = (close - open) / (high - low + 0.001)
    df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
    
    # Volume normalization
    df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
    df['volume_normalized'] = df['volume'] / (df['volume_5d_avg'] + 1e-8)
    
    # Adaptive signal based on volatility clustering strength
    strong_clustering_mask = df['volatility_autocorr'] > df['volatility_autocorr'].rolling(window=20).quantile(0.7)
    weak_clustering_mask = df['volatility_autocorr'] <= df['volatility_autocorr'].rolling(window=20).quantile(0.7)
    
    # Initialize factor
    df['factor'] = np.nan
    
    # Strong clustering: Momentum decay rate * Price efficiency
    df.loc[strong_clustering_mask, 'factor'] = (
        df.loc[strong_clustering_mask, 'momentum_decay'] * 
        df.loc[strong_clustering_mask, 'price_efficiency']
    )
    
    # Weak clustering: Momentum decay rate * (volume / volume_5d_avg)
    df.loc[weak_clustering_mask, 'factor'] = (
        df.loc[weak_clustering_mask, 'momentum_decay'] * 
        df.loc[weak_clustering_mask, 'volume_normalized']
    )
    
    return df['factor']
