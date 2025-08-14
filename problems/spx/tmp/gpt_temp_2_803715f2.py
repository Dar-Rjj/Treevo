import pandas as pd
import pandas as pd

def heuristics(df, n=20, m=10, p=14, q=10, volume_threshold=2.0, volatility_threshold=2.0, range_threshold=1.5):
    # Calculate Price Momentum
    df['price_momentum'] = (df['close'] / df['close'].shift(n) - 1) * 100
    
    # Identify Volume Spikes
    df['avg_volume'] = df['volume'].rolling(window=m).mean()
    df['volume_spike'] = (df['volume'] / df['avg_volume']) > volume_threshold
    df['adjusted_momentum'] = df['price_momentum']
    df.loc[df['volume_spike'], 'adjusted_momentum'] *= 1.5  # Scaling factor for volume spike
    
    # Integrate Price Volatility
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['avg_true_range'] = df['true_range'].rolling(window=p).mean()
    df['volatility_spike'] = (df['true_range'] / df['avg_true_range']) > volatility_threshold
    df.loc[df['volatility_spike'], 'adjusted_momentum'] *= 1.2  # Different scaling factor for volatility spike
    
    # Incorporate Trading Range
    df['trading_range'] = df['high'] - df['low']
    df['avg_trading_range'] = df['trading_range'].rolling(window=q).mean()
    df['wide_trading_range'] = (df['trading_range'] / df['avg_trading_range']) > range_threshold
    df.loc[df['wide_trading_range'], 'adjusted_momentum'] *= 1.3  # Another scaling factor for wide trading range
    
    return df['adjusted_momentum']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor = heuristics(df)
# print(factor)
