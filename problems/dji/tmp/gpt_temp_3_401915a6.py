import pandas as pd
import pandas as pd

def heuristics_v2(df, n_days):
    # Calculate Daily High-to-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate Intraday High-Low Spread (same as high_low_range)
    df['intraday_spread'] = df['high_low_range']
    
    # Calculate Open-Close Direction
    df['open_close_direction'] = (df['open'] > df['close']).astype(int)
    
    # Calculate Average High-to-Low Range over N Days
    df['avg_high_low_range'] = df['high_low_range'].rolling(window=n_days).mean()
    
    # Compare with Current High-to-Low Range
    df['momentum_signal'] = (df['high_low_range'] > df['avg_high_low_range']).astype(int)
    
    # Integrate Volume for Price Movement
    df['volume_weighted_spread'] = df['volume'] * df['intraday_spread']
    
    # Calculate Final Factor
    df['weighted_spread_sum'] = df['volume_weighted_spread'].rolling(window=n_days).sum()
    df['final_factor'] = df.apply(lambda row: -row['weighted_spread_sum'] if row['open_close_direction'] == 1 else row['weighted_spread_sum'], axis=1)
    
    # Combine with Momentum Signal
    df['final_factor'] = df.apply(lambda row: row['final_factor'] * 1.5 if row['momentum_signal'] == 1 else row['final_factor'], axis=1)
    
    return df['final_factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [...],
#     'high': [...],
#     'low': [...],
#     'close': [...],
#     'volume': [...]
# }, index=pd.to_datetime([...]))
# factor_values = heuristics_v2(df, n_days=10)
