import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Generate a novel and interpretable alpha factor based on the given price and volume data.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume'] indexed by date.
    
    Returns:
    pd.Series: Series of alpha factor values indexed by date.
    """
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the intraday range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate the relative strength index (RSI) over a 14-day window
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate the average true range (ATR) over a 14-day window
    tr = df[['high' - 'low', ('high' - 'close').abs(), ('low' - 'close').abs()]].max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Calculate the money flow multiplier
    df['money_flow_multiplier'] = (df['close'] - df['low']) - (df['high'] - df['close']) / (df['high'] - df['low'])
    
    # Calculate the money flow volume
    df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
    
    # Calculate the money flow ratio
    df['money_flow_ratio'] = df['money_flow_volume'].rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    
    # Combine the factors into a single alpha factor
    df['alpha_factor'] = (df['rsi'] - 50) * df['atr'] * df['money_flow_ratio']
    
    # Return the alpha factor series
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
