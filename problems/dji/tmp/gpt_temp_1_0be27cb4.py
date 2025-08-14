import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    def std_dev(price, periods=20):
        return price.rolling(window=periods).std()

    def momentum(volume, periods=10):
        return volume.pct_change(periods=periods)

    std_close = std_dev(df['close'])
    vol_momentum = momentum(df['volume'])
    combined_factor = (std_close + vol_momentum).rename('combined_factor').dropna()
    
    X = combined_factor.values.reshape(-1, 1)
    y = df['close'].pct_change().shift(-1).loc[combined_factor.index].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    heuristics_matrix = pd.Series(model.predict(X).flatten(), index=combined_factor.index, name='heuristic_factor')
    
    return heuristics_matrix
