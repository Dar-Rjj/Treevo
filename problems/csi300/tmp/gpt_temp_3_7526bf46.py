import pandas as pd
import pandas as pd

def heuristics_v2(df, n=14):
    # Compute Raw Momentum
    df['raw_momentum'] = df['close'].pct_change(periods=n)
    
    # Weight by Cumulative Volume
    df['cumulative_volume'] = df['volume'].rolling(window=n).sum()
    df['volume_weighted_momentum'] = df['raw_momentum'] / df['cumulative_volume']
    
    # Calculate True Range (TR)
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    
    # Calculate Average True Range (ATR)
    df['atr'] = df['tr'].rolling(window=n).mean()
    
    # Calculate Positive Directional Movement (+DM)
    df['pos_dm'] = (df['high'] - df['high'].shift(1)).apply(lambda x: max(x, 0) if x > 0 and df['low'] < df['low'].shift(1) else 0)
    
    # Calculate Negative Directional Movement (-DM)
    df['neg_dm'] = (df['low'].shift(1) - df['low']).apply(lambda x: max(x, 0) if x > 0 and df['high'] > df['high'].shift(1) else 0)
    
    # Smooth +DM and -DM over n periods
    df['pos_dm_smoothed'] = df['pos_dm'].rolling(window=n).mean()
    df['neg_dm_smoothed'] = df['neg_dm'].rolling(window=n).mean()
    
    # Calculate +DI and -DI
    df['pos_di'] = df['pos_dm_smoothed'] / df['atr']
    df['neg_di'] = df['neg_dm_smoothed'] / df['atr']
    
    # Calculate ADMI
    df['admi'] = (df['pos_di'] - df['neg_di']) / (df['pos_di'] + df['neg_di'])
    
    # Combine Volume-Weighted Momentum with ADMI
    df['factor'] = df['volume_weighted_momentum'] * df['admi'] + df['volume_weighted_momentum']
    
    return df['factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [...],
#     'high': [...],
#     'low': [...],
#     'close': [...],
#     'amount': [...],
#     'volume': [...]
# })
# factor_values = heuristics_v2(df)
