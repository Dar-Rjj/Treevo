import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10, m=14):
    # Calculate Daily Return
    df['Daily_Return'] = df['close'].pct_change()
    
    # Calculate Volume Change Ratio
    df['Volume_Change_Ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Compute Weighted Momentum
    df['Weighted_Momentum'] = (df['Daily_Return'] * df['Volume_Change_Ratio']).rolling(window=n).sum()
    
    # Calculate True Range
    df['True_Range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1
    )
    
    # Calculate Average True Range (ATR)
    df['ATR'] = df['True_Range'].rolling(window=m).mean()
    
    # Calculate Enhanced ATR
    df['Enhanced_ATR'] = df['ATR'] * (1 + 0.5 * (df['high'] - df['low']) / df['close'].shift(1))
    
    # Adjust for Price Volatility
    df['Vol_Adjusted_Momentum'] = df['Weighted_Momentum'] - df['Enhanced_ATR']
    
    return df['Vol_Adjusted_Momentum']
