import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['open']
    
    # Calculate Volume-Weighted Intraday Return
    df['vol_weighted_intraday_return'] = df['intraday_return'] * df['volume']
    
    # Adaptive Exponential Moving Average of Volume-Weighted Intraday Return
    def adaptive_ema(series, alpha):
        ema = series.ewm(alpha=alpha, adjust=False).mean()
        return ema
    
    # Calculate recent volatility
    df['intraday_volatility'] = df['intraday_return'].rolling(window=20).std() * np.sqrt(20)
    recent_volatility = df['intraday_volatility'].rolling(window=10).mean()
    vol_adjustment = 1 + (recent_volatility - recent_volatility.mean()) / recent_volatility.std()
    dynamic_alpha = 2 / (30 + vol_adjustment)

    df['adaptive_ema_vol_weighted_intraday_return'] = adaptive_ema(df['vol_weighted_intraday_return'], dynamic_alpha)
    
    # Simple Moving Average of Intraday Volatility
    df['sma_intraday_volatility'] = df['intraday_volatility'].rolling(window=20).mean()
    
    # Simple Moving Average of Recent Volatility
    df['sma_recent_volatility'] = recent_volatility.rolling(window=10).mean()
    
    # Volume-Weighted Intraday Momentum
    df['vol_weighted_intraday_momentum'] = df['intraday_return'] * df['volume']
    
    # Adaptive Exponential Moving Average of Volume-Weighted Intraday Momentum
    df['adaptive_ema_vol_weighted_intraday_momentum'] = adaptive_ema(df['vol_weighted_intraday_momentum'], dynamic_alpha)
    
    # Combine Factors
    df['final_factor'] = (
        df['adaptive_ema_vol_weighted_intraday_return'] +
        df['sma_intraday_volatility'] +
        df['sma_recent_volatility']
    )
    
    return df['final_factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 102, 101, ...],
#     'high': [105, 107, 106, ...],
#     'low': [98, 99, 97, ...],
#     'close': [103, 104, 100, ...],
#     'amount': [10000, 12000, 11000, ...],
#     'volume': [1000, 1200, 1100, ...]
# }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', ...]))

# factor = heuristics_v2(df)
