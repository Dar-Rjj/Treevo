import pandas as pd
import pandas as pd

def heuristics_v2(df, N=20, M=10):
    # Calculate Daily Close-to-Close Return
    df['close_to_close_return'] = df['close'].pct_change()

    # Calculate Accumulated Volume Trend (AVT)
    df['volume_diff'] = df['volume'].diff()
    df['AVT'] = 0.0  # Initialize AVT to 0
    for i in range(1, N+1):
        if i == 1:
            df.loc[df.index[i:], 'AVT'] += df.loc[df.index[i:], 'volume_diff']
        else:
            df.loc[df.index[i:], 'AVT'] += (df.loc[df.index[i:], 'volume_diff'] * (df.loc[df.index[i-1:i], 'AVT'] > 0).astype(int)) - (df.loc[df.index[i:], 'volume_diff'] * (df.loc[df.index[i-1:i], 'AVT'] <= 0).astype(int))

    # Calculate Weighted Close-to-Close Return
    df['weighted_return'] = df['close_to_close_return'] * (1 + df['AVT'].abs())

    # Incorporate Price Momentum
    df['avg_momentum'] = df['close_to_close_return'].rolling(window=M).mean()
    df['final_factor'] = df['weighted_return'] + df['avg_momentum']

    # Determine Sign of AVT and adjust final factor
    df['final_factor'] = df.apply(lambda row: row['final_factor'] * (1 + row['AVT']) if row['AVT'] > 0 else row['final_factor'] * (1 - row['AVT']), axis=1)

    return df['final_factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [...],
#     'high': [...],
#     'low': [...],
#     'close': [...],
#     'amount': [...],
#     'volume': [...]
# }, index=pd.to_datetime([...]))
# alpha = heuristics_v2(df)
