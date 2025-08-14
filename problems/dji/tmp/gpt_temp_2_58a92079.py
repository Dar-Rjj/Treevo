import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Generate a novel and interpretable alpha factor from the given market data.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns 'open', 'high', 'low', 'close', and 'volume'.
    
    Returns:
    pd.Series: A Series representing the factor values indexed by date.
    """
    # Calculate the price range
    price_range = df['high'] - df['low']
    
    # Calculate the average true range
    df['true_range'] = df[['high' - 'low', abs('high' - df['close'].shift(1)), abs('low' - df['close'].shift(1))]].max(axis=1)
    atr = df['true_range'].rolling(window=14).mean()
    
    # Calculate the normalized volume
    avg_volume = df['volume'].rolling(window=30).mean()
    norm_volume = df['volume'] / avg_volume
    
    # Calculate the normalized price change
    price_change = df['close'] - df['open']
    norm_price_change = price_change / price_range
    
    # Combine the factors into a single alpha factor
    alpha_factor = (norm_price_change * norm_volume) / atr
    
    return alpha_factor
