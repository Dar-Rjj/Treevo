import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the difference between close and open prices
    diff_close_open = df['close'] - df['open']
    
    # Calculate the average true range as a measure of volatility
    df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Calculate the volume weighted price
    vwp = (df['volume'] * df['close']).rolling(window=5).mean() / df['volume'].rolling(window=5).mean()
    
    # Generate the factor as the ratio of the difference in close and open, adjusted by the average true range,
    # and further weighted by the ratio of the current closing price to the volume weighted price
    factor = diff_close_open / (df['atr_14'] + 1e-7) * (df['close'] / vwp)
    
    return factor
