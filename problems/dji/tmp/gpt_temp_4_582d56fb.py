import pandas as pd
import pandas as pd

def heuristics_v2(df, window_size=20):
    # Calculate Daily Return
    df['daily_return'] = df['close'] - df['close'].shift(1)
    
    # Calculate Volume Change
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Weighted Price Momentum
    df['weighted_momentum'] = df['daily_return'] * df['volume']
    df['weighted_momentum_avg'] = df['weighted_momentum'].rolling(window=window_size).mean()
    
    # Relative Strength
    high_low_range = df['high'].rolling(window=window_size).max() - df['low'].rolling(window=window_size).min()
    lowest_low = df['low'].rolling(window=window_size).min()
    relative_strength = (df['high'] - lowest_low) / high_low_range
    
    # Intraday Volatility
    df['true_range'] = df[['high', 'low', df['close'].shift(1)]].max(axis=1) - df[['high', 'low', df['close'].shift(1)]].min(axis=1)
    intraday_volatility = df['true_range'].rolling(window=window_size).mean()
    
    # Adjusted VPMI
    vpmi = df['weighted_momentum_avg'] * relative_strength
    
    # Intraday Volatility-Adjusted Momentum
    iv_adjusted_momentum = vpmi / intraday_volatility
    
    # Final Alpha Factor
    final_alpha_factor = vpmi + iv_adjusted_momentum
    
    return final_alpha_factor

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
