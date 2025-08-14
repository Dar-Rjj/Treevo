import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Generates a novel and interpretable alpha factor using open, high, low, close, volume data.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] indexed by date.
    
    Returns:
    pd.Series: Alpha factor values indexed by date.
    """
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the intraday range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate the relative strength index (RSI) for a 14-day window
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate the average true range (ATR) for a 14-day window
    df['tr'] = df[['high' - 'low', 'high' - 'close'].shift(1), 'close'.shift(1) - 'low']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Calculate the money flow multiplier
    df['mfm'] = (df['close'] - df['low']) - (df['high'] - df['close']) / (df['high'] - df['low'])
    
    # Calculate the money flow volume
    df['mfv'] = df['mfm'] * df['volume']
    
    # Calculate the money flow index (MFI) for a 14-day window
    df['positive_mfv'] = df['mfv'].where(df['mfv'] > 0, 0)
    df['negative_mfv'] = df['mfv'].where(df['mfv'] < 0, 0)
    df['mfi'] = 100 - (100 / (1 + (df['positive_mfv'].rolling(window=14).sum() / df['negative_mfv'].rolling(window=14).sum().abs())))
    
    # Combine the factors into a single alpha factor
    alpha_factor = (df['rsi'] / 100) * (df['atr'] / df['intraday_range']) * (df['mfi'] / 100)
    
    return alpha_factor

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
