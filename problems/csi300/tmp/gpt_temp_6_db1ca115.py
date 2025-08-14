import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Dynamic Volatility Measure
    df['daily_range'] = df['high'] - df['low']
    df['std_dev_close'] = df['close'].rolling(window=20).std()
    df['combined_volatility'] = 0.5 * df['daily_range'] + 0.5 * df['std_dev_close']
    
    # Adaptive Window Calculation
    def adjust_window_size(volatility, trend):
        base_window = 20
        if volatility > df['combined_volatility'].median():
            window = base_window - 3
        else:
            window = base_window + 3
        
        if trend > 0:
            window -= 1
        elif trend < 0:
            window += 1
        
        return max(5, min(window, 30))  # Limit the window between 5 and 30
    
    df['vol_trend'] = df['volume'].rolling(window=20).mean().pct_change()
    df['adaptive_window'] = df.apply(lambda row: adjust_window_size(row['combined_volatility'], row['vol_trend']), axis=1)
    
    # Rolling Statistics with Adaptive Window
    volume_weighted_rolling_mean = []
    volume_weighted_rolling_std = []
    
    for i in range(len(df)):
        window = int(df['adaptive_window'].iloc[i])
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        sub_df = df.iloc[start_idx:end_idx]
        
        rolling_mean = sub_df['volume_weighted_return'].mean()
        rolling_std = sub_df['volume_weighted_return'].std()
        
        volume_weighted_rolling_mean.append(rolling_mean)
        volume_weighted_rolling_std.append(rolling_std)
    
    df['factor_value'] = (pd.Series(volume_weighted_rolling_mean, index=df.index) /
                          pd.Series(volume_weighted_rolling_std, index=df.index))
    
    return df['factor_value']
