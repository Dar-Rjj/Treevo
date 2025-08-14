import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Define weights for recent and older data
    recent_weights = {'intraday_range': 0.7, 'close_to_open_return': 0.3}
    older_weights = {'intraday_range': 0.5, 'close_to_open_return': 0.5}
    
    # Apply the weighted combination
    df['weighted_intraday_volatility_adjusted_return'] = (
        df['intraday_range'] * recent_weights['intraday_range'] + 
        df['close_to_open_return'] * recent_weights['close_to_open_return']
    )
    
    # For older data, use the older weights
    df['weighted_intraday_volatility_adjusted_return'] = (
        df['weighted_intraday_volatility_adjusted_return']
        .where(df.index >= df.index.max() - pd.Timedelta(days=30),  # Adjust the 30 days threshold as needed
               df['intraday_range'] * older_weights['intraday_range'] + 
               df['close_to_open_return'] * older_weights['close_to_open_return'])
    )
    
    return df['weighted_intraday_volatility_adjusted_return']

# Example usage:
# df = pd.DataFrame(...)  # Your DataFrame here
# factor_values = heuristics_v2(df)
