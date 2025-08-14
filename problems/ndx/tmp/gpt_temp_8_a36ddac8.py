import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Generate a novel and interpretable alpha factor to predict future stock returns.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns 'open', 'high', 'low', 'close', 'volume' indexed by date.
    
    Returns:
    pd.Series: Alpha factor values indexed by date.
    """
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the intraday range
    df['intraday_range'] = (df['high'] - df['low']) / df['low']
    
    # Calculate the relative strength index (RSI) over a 14-day period
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate the average true range (ATR) over a 14-day period
    df['tr'] = df[['high' - 'low', 
                   ('high' - df['close'].shift(1)).abs(), 
                   ('low' - df['close'].shift(1)).abs()]].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Combine the factors into a single alpha factor
    df['alpha_factor'] = (df['intraday_range'] * df['rsi'] / df['atr'])
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
