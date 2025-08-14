import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=21, m=21):
    # Calculate Price Momentum
    close_trend = df['close'].pct_change()
    close_ema = close_trend.ewm(span=n, min_periods=n).mean()
    momentum_score = close_ema.diff().rolling(window=n, min_periods=n).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)))
    
    # Volume Confirmation
    volume_trend = df['volume'].pct_change()
    volume_dema = 2 * volume_trend.ewm(span=m, min_periods=m).mean() - volume_trend.ewm(span=2*m, min_periods=2*m).mean()
    volume_score = volume_dema.diff().rolling(window=m, min_periods=m).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)))
    
    # Enhanced Volatility Analysis
    high_low_range = df['high'] - df['low']
    open_close_range = (df['open'] - df['close']).abs()
    combined_range = pd.concat([high_low_range, open_close_range], axis=1).max(axis=1)
    volatility_dema = 2 * combined_range.ewm(span=n, min_periods=n).mean() - combined_range.ewm(span=2*n, min_periods=2*n).mean()
    volatility_score = volatility_dema.diff().rolling(window=n, min_periods=n).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)))
    
    # Combine Momentum, Volume, and Enhanced Volatility Scores
    combined_score = (momentum_score * volume_score) / np.sqrt(volatility_score)
    
    return combined_score.dropna()

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 101, 99, 102, 103],
#     'high': [102, 103, 101, 104, 105],
#     'low': [98, 99, 97, 100, 101],
#     'close': [101, 100, 102, 103, 104],
#     'volume': [1000, 1500, 1200, 1300, 1400]
# })
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
