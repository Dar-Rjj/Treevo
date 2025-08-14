import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Average Price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Compute Daily Change in Volume-Weighted Price
    df['vwap_change'] = df['vwap'].diff()
    
    # Separate Positive and Negative Changes
    df['positive_change'] = df['vwap_change'].apply(lambda x: max(x, 0))
    df['negative_change'] = df['vwap_change'].apply(lambda x: abs(min(x, 0)))
    
    # Compute 14-Day Averages for RSI
    df['avg_positive_14'] = df['positive_change'].rolling(window=14).mean()
    df['avg_negative_14'] = df['negative_change'].rolling(window=14).mean()
    
    # Calculate RSI
    df['relative_strength'] = df['avg_positive_14'] / df['avg_negative_14']
    df['rsi'] = 100 - (100 / (1 + df['relative_strength']))
    
    # Intraday Range Growth and Volume-Weighted Moving Average
    df['intraday_range'] = df['high'] - df['low']
    df['prev_intraday_range'] = df['intraday_range'].shift(1)
    df['intraday_range_growth'] = (df['intraday_range'] - df['prev_intraday_range']) / df['prev_intraday_range']
    
    # Volume-Weighted Moving Average
    df['volume_weighted_close'] = df['close'] * df['volume']
    df['vwap_sma'] = df['volume_weighted_close'].rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    
    # Long-Term and Short-Term Volume-Weighted Average Return
    df['daily_return'] = df['close'].pct_change()
    
    # Long-Term Volume-Weighted Average Return (Momentum Component)
    df['long_term_vol_weighted_return'] = (df['daily_return'].rolling(window=100) * df['volume']).rolling(window=100).sum() / df['volume'].rolling(window=100).sum()
    
    # Short-Term Volume-Weighted Average Return (Reversal Component)
    df['short_term_vol_weighted_return'] = (df['daily_return'].rolling(window=5) * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    
    # Combine All Components
    df['final_alpha_factor'] = df['rsi'] + df['intraday_range_growth'] + df['vwap_sma'] + df['long_term_vol_weighted_return'] - df['short_term_vol_weighted_return']
    
    return df['final_alpha_factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
