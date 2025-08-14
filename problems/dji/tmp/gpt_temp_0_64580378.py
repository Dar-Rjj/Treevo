import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Generate a novel and interpretable alpha factor using (open, high, low, close) prices and volume.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] and indexed by date.
    
    Returns:
    pd.Series: A Series representing the alpha factor values, indexed by date.
    """
    # Calculate the relative strength of the closing price
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff(1).rolling(window=14, min_periods=1).mean() / 
                                 df['close'].diff(1).abs().rolling(window=14, min_periods=1).mean())))
    
    # Calculate the percentage change in volume
    df['volume_change'] = df['volume'].pct_change()
    
    # Calculate the average true range
    df['tr'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['close'].shift(1) - 'low']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14, min_periods=1).mean()
    
    # Calculate the price momentum
    df['momentum'] = df['close'] / df['close'].shift(20) - 1
    
    # Combine the factors into a single alpha factor
    df['alpha_factor'] = (df['rsi'] + df['volume_change'] + df['atr'] + df['momentum']) / 4
    
    return df['alpha_factor']
