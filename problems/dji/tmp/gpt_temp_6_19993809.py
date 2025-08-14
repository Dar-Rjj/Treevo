import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Log Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate 20-Day Moving Average of Close Price
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate 20-Day Standard Deviation of Log Returns
    df['std_20'] = df['log_return'].rolling(window=20).std()
    
    # Calculate Trend Momentum Indicator
    df['trend_momentum'] = (df['close'] - df['ma_20']) / df['std_20']
    
    # Calculate Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume-Averaged Price
    df['volume_averaged_price'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate Volume-Averaged Momentum
    n_periods = 5
    df['volume_averaged_momentum'] = df['volume_averaged_price'].diff(n_periods)
    
    # Calculate True Range
    df['true_range'] = df[['high', 'low', df['close'].shift(1)]].max(axis=1) - df[['high', 'low', df['close'].shift(1)]].min(axis=1)
    
    # Calculate 5-Day Intraday Volatility
    df['intraday_volatility'] = df['intraday_high_low_spread'].abs().rolling(window=5).sum()
    
    # Calculate Intraday Stability
    df['intraday_stability'] = df['intraday_high_low_spread'] / df['intraday_volatility']
    df['intraday_stability'] = 1 / df['intraday_stability']
    
    # Combine Trend and Intraday Components
    df['combined_trend_intraday'] = (df['intraday_high_low_spread'] * df['close_to_open_return']) / df['std_20']
    
    # Volume-Adjusted Momentum
    df['volume_adjusted_momentum'] = df['volume_averaged_momentum'] / df['true_range']
    
    # Confirm Momentum with Volume Surge
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['normalized_volume_change'] = df['volume_change'] / df['volume'].shift(1)
    df['normalized_volume_change'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['normalized_volume_change'].fillna(0, inplace=True)
    
    volume_change_threshold = 0.1
    df['volume_surge'] = df['normalized_volume_change'] > volume_change_threshold
    df['combined_momentum'] = df['volume_adjusted_momentum'] * df['normalized_volume_change']
    df['combined_momentum'] = df['combined_momentum'].where(df['volume_surge'], 0)
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['combined_trend_intraday'] + df['combined_momentum']
    
    # Adjust for Long-Term Reversal
    df['long_term_smoothed_return'] = df['close'].pct_change(252).rolling(window=252).mean()
    df['final_alpha_factor'] -= df['long_term_smoothed_return']
    
    return df['final_alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
