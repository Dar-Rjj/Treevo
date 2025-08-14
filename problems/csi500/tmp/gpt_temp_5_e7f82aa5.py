import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['high'] - df['low']

    # Calculate Volume-Weighted Intraday Return
    volume_weighted_intraday_return = intraday_return * df['volume']

    # Calculate Exponential Moving Average of Volume-Weighted Intraday Return
    ema_vol_weighted_intraday_return = volume_weighted_intraday_return.ewm(span=20, adjust=False, alpha=0.15).mean()

    # Calculate Intraday Volatility
    squared_intraday_return = intraday_return ** 2
    intraday_volatility = np.sqrt(squared_intraday_return.rolling(window=20).sum())

    # Calculate Simple Moving Average of Intraday Volatility
    sma_intraday_volatility = intraday_volatility.rolling(window=20).mean()

    # Adjust for Recent Volatility
    recent_volatility_sma = intraday_volatility.rolling(window=5).mean()

    # Combine Factors
    factor = ema_vol_weighted_intraday_return + sma_intraday_volatility - recent_volatility_sma

    return factor

# Example usage:
# df = pd.DataFrame({
#     'open': [...],
#     'high': [...],
#     'low': [...],
#     'close': [...],
#     'amount': [...],
#     'volume': [...]
# }, index=pd.to_datetime([...]))

# factor_values = heuristics_v2(df)
# print(factor_values)
