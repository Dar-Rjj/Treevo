import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Efficiency Ratio
    price_range = data['high'] - data['low']
    directional_change = np.abs(data['close'] - data['open'])
    efficiency_ratio = directional_change / (price_range + 1e-8)
    efficiency_ratio = np.arcsinh(efficiency_ratio * 100)  # Inverse hyperbolic sine
    
    # Volume Clustering Strength
    # Volume concentration
    volume_10d_percentile = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: np.percentile(x, 70) if len(x) >= 5 else np.nan, raw=True
    )
    volume_concentration = data['volume'] / (volume_10d_percentile + 1e-8)
    volume_concentration = 1 / (1 + np.exp(-volume_concentration))  # Logistic function
    
    # Volume autocorrelation
    volume_autocorr = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=5) if len(x) >= 10 else np.nan, raw=False
    )
    volume_clustering = volume_concentration * np.abs(volume_autocorr)
    
    # Momentum Persistence Signal
    momentum_signal = efficiency_ratio * volume_clustering
    momentum_signal = np.sign(momentum_signal) * np.abs(momentum_signal) ** (1/3)  # Cubic root
    
    # Multi-timeframe factor
    short_ma = momentum_signal.ewm(span=3, min_periods=2).mean()
    medium_ma = momentum_signal.rolling(window=8, min_periods=5).mean()
    factor = short_ma / (medium_ma + 1e-8)
    
    # Handle NaN values
    factor = factor.fillna(0)
    
    return factor
