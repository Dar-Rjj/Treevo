import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the daily return
    df['return'] = df['close'].pct_change()
    
    # Calculate the signed volume: If today's close is higher than yesterday's, volume is positive, otherwise negative
    df['signed_volume'] = df['volume'] * (df['close'].diff() > 0).astype(int)
    df['signed_volume'] = df['signed_volume'].replace({0: -df['volume']})
    
    # Calculate the money flow: (high + low) / 2 * signed_volume
    df['money_flow'] = ((df['high'] + df['low']) / 2) * df['signed_volume']
    
    # Calculate the money flow factor: money_flow / sum of absolute money_flow over the last 14 days
    df['money_flow_factor'] = df['money_flow'] / df['money_flow'].abs().rolling(window=14).sum()
    
    # The alpha factor is a combination of the daily return and the money flow factor
    df['alpha_factor'] = (df['return'].shift(1) * df['money_flow_factor']).fillna(0)
    
    # Return the alpha factor as a Series
    return df['alpha_factor']
