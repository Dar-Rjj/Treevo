import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
    
    # Weighted Sum of Intraday and Close-to-Open Returns
    weighted_intraday = intraday_volatility.abs() * intraday_volatility
    weighted_close_to_open = close_to_open_return.abs() * close_to_open_return
    preliminary_factor = weighted_intraday + weighted_close_to_open
    
    # Calculate True Range (TR)
    df['previous_close'] = df['close'].shift(1)
    true_range = df[['high' - 'low', 
                     (df['high'] - df['previous_close']).abs(), 
                     (df['low'] - df['previous_close']).abs()]].max(axis=1)
    
    # Calculate Positive Directional Movement (+DM)
    positive_dm = (df['high'].diff().clip(lower=0) - df['low'].diff().clip(upper=0)).clip(lower=0)
    
    # Calculate Negative Directional Movement (-DM)
    negative_dm = (df['low'].diff().clip(upper=0) - df['high'].diff().clip(lower=0)).clip(upper=0)
    
    # Smooth +DM and -DM over 14 periods
    smooth_positive_dm = positive_dm.rolling(window=14).sum()
    smooth_negative_dm = negative_dm.rolling(window=14).sum()
    tr_14 = true_range.rolling(window=14).sum()
    
    # Calculate +DI and -DI
    plus_di = smooth_positive_dm / tr_14
    minus_di = smooth_negative_dm / tr_14
    
    # Calculate ADMI
    admi = (plus_di - minus_di) / (plus_di + minus_di)
    
    # Multiply Preliminary Factor by ADMI
    alpha_factor = preliminary_factor * admi
    
    return alpha_factor
