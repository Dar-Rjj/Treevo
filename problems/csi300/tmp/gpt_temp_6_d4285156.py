import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Component with Exponential Decay
    returns = df['close'].pct_change()
    decayed_momentum = returns.rolling(window=10, min_periods=1).apply(
        lambda x: np.average(x, weights=np.power(0.94, np.arange(len(x))[::-1]))
    )
    
    # Volume-Weighted Price Action
    price_changes = df['close'].diff()
    volume_weighted_changes = price_changes * df['volume']
    
    # Divergence Detection
    rolling_corr = decayed_momentum.rolling(window=5).corr(volume_weighted_changes)
    historical_median = rolling_corr.rolling(window=20).median()
    divergence = (rolling_corr - historical_median).abs()
    
    # Signal Integration with Linear Time Decay
    raw_signal = decayed_momentum * divergence
    time_weights = np.linspace(0.1, 1.0, len(raw_signal))
    weighted_signal = raw_signal * time_weights
    
    # Volatility Scaling
    volatility = returns.rolling(window=10, min_periods=1).std()
    final_signal = weighted_signal / volatility.replace(0, np.nan)
    
    return final_signal
