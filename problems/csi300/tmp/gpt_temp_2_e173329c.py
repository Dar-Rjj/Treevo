import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, lookback_period=10):
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Calculate Volume-Adjusted Typical Price (VATP)
    df['typical_price'] = (df['high'] + df['low'] + 2 * df['close']) / 4
    df['volume_adjusted_typical_price'] = df['typical_price'] * np.sqrt(df['volume'])
    total_vatp = df['volume_adjusted_typical_price'].sum()
    total_volume = df['volume'].sum()
    vatp = total_vatp / total_volume
    
    # Calculate Momentum
    df['momentum'] = (df['close'] - df['close'].shift(lookback_period)) / df['close'].shift(lookback_period)
    
    # Adjust for Volume Changes
    df['volume_ratio'] = df['volume'] / df['volume'].shift(lookback_period)
    df['adjusted_momentum'] = df['momentum'] + (df['momentum'] * df['volume_ratio'])
    
    # Combine Intraday Volatility and VATP
    df['intraday_volatility_vatp_high_diff'] = df['high'] - vatp
    df['intraday_volatility_vatp_low_diff'] = vatp - df['low']
    df['intraday_volatility_vatp_combined'] = df['intraday_volatility_vatp_high_diff'] + df['intraday_volatility_vatp_low_diff']
    
    # Apply a weighted moving average to smooth the factor
    weights = df['volume'].rolling(window=lookback_period).apply(lambda x: x / x.sum(), raw=False)
    df['weighted_moving_average'] = (df['intraday_volatility_vatp_combined'] * weights).rolling(window=lookback_period).sum()
    
    return df['weighted_moving_average']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 105, 108, ...],
#     'high': [110, 115, 118, ...],
#     'low': [98, 100, 102, ...],
#     'close': [106, 110, 112, ...],
#     'amount': [1000, 1050, 1080, ...],
#     'volume': [100, 105, 108, ...]
# }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', ...]))

# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
