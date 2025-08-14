import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Define the base lookback period for standard deviation
    base_lookback = 20
    
    # Calculate rolling standard deviation of close prices
    df['std_close'] = df['close'].rolling(window=base_lookback).std()
    
    # Adjust the lookback period based on market volatility
    def adjust_lookback(std, high_vol_threshold, low_vol_threshold, high_vol_lookback, low_vol_lookback):
        if std > high_vol_threshold:
            return high_vol_lookback
        elif std < low_vol_threshold:
            return low_vol_lookback
        else:
            return base_lookback

    high_vol_threshold = df['std_close'].quantile(0.75)
    low_vol_threshold = df['std_close'].quantile(0.25)
    high_vol_lookback = 10
    low_vol_lookback = 30
    
    df['lookback'] = df['std_close'].apply(lambda x: adjust_lookback(x, high_vol_threshold, low_vol_threshold, high_vol_lookback, low_vol_lookback))
    
    # Calculate the dynamic simple moving average (SMA) of close prices
    df['sma_close'] = df.apply(lambda row: df.loc[:row.name, 'close'].rolling(window=row['lookback']).mean().iloc[-1], axis=1)
    
    # Calculate true range
    df['true_range'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1).shift(1)
    
    # Apply volume weighting to true range
    df['volume_weighted_tr'] = df['volume'] * df['true_range']
    
    # Calculate the rolling average of the volume-weighted true ranges
    df['avg_volume_weighted_tr'] = df['volume_weighted_tr'].rolling(window=df['lookback']).mean()
    
    # Compute price momentum
    n = df['lookback']
    df['price_momentum'] = (df['close'] - df['sma_close']) / df['close'].rolling(window=n).sum()
    
    # Final alpha factor: price momentum divided by volume-adjusted volatility
    df['alpha_factor'] = df['price_momentum'] / df['avg_volume_weighted_tr']
    
    return df['alpha_factor']

# Example usage:
# df = pd.DataFrame(...)  # Assume df is a DataFrame with columns: 'date', 'open', 'high', 'low', 'close', 'amount', 'volume'
# alpha_factor = heuristics_v2(df)
