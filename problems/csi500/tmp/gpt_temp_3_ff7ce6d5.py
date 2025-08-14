import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10, spike_factor=2):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Identify Volume Spike
    df['volume_change'] = df['volume'].diff()
    df['volume_spike'] = (df['volume_change'] > df['volume'].shift(1) * spike_factor).astype(int)
    
    # Adjust Momentum by Volume Spike
    df['n_day_momentum'] = df['daily_return'].rolling(window=n).sum()
    df['adjusted_momentum'] = df['n_day_momentun'] * (spike_factor if df['volume_spike'] else 1)
    
    # Calculate Daily Price Movement and Sign
    df['price_movement'] = df['close'] - df['open']
    df['sign'] = (df['close'] > df['open']).astype(int) * 2 - 1
    
    # Determine Adjusted Volume
    df['adjusted_volume'] = df['volume'] * df['sign']
    
    # Create Weighted Average
    df['weighted_price_movement'] = df['price_movement'] * df['adjusted_volume']
    df['sum_weighted_movement'] = df['weighted_price_movement'].rolling(window=n).sum()
    df['sum_adjusted_volume'] = df['adjusted_volume'].rolling(window=n).sum()
    df['weighted_average'] = df['sum_weighted_movement'] / df['sum_adjusted_volume']
    
    # Return the final alpha factor
    return df['weighted_average'].dropna()

# Example usage:
# df = pd.DataFrame({
#     'open': [10, 11, 12, 13, 14],
#     'high': [15, 16, 17, 18, 19],
#     'low': [8, 9, 10, 11, 12],
#     'close': [12, 13, 14, 15, 16],
#     'amount': [1000, 1100, 1200, 1300, 1400],
#     'volume': [100, 150, 200, 250, 300]
# })
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
