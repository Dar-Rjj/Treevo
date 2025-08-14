import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Momentum
    intraday_momentum = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
    
    # Weighted Sum of Intraday and Close-to-Open Returns
    weight_intraday = abs(intraday_momentum)
    weight_close_to_open = abs(close_to_open_return)
    preliminary_factor = (intraday_momentum * weight_intraday) + (close_to_open_return * weight_close_to_open)
    
    # Calculate True Range (TR)
    df['previous_close'] = df['close'].shift(1)
    true_range = df.apply(lambda row: max(row['high'] - row['low'], 
                                         abs(row['high'] - row['previous_close']), 
                                         abs(row['low'] - row['previous_close'])), axis=1)
    
    # Calculate Positive Directional Movement (+DM)
    df['previous_high'] = df['high'].shift(1)
    df['previous_low'] = df['low'].shift(1)
    positive_dm = df.apply(lambda row: row['high'] - row['previous_high'] if row['high'] > row['previous_high'] and (row['high'] - row['previous_high']) > (row['previous_low'] - row['low']) else 0, axis=1)
    
    # Calculate Negative Directional Movement (-DM)
    negative_dm = df.apply(lambda row: row['previous_low'] - row['low'] if row['previous_low'] > row['low'] and (row['previous_low'] - row['low']) > (row['high'] - row['previous_high']) else 0, axis=1)
    
    # Smooth +DM and -DM over a 14-day period
    smooth_positive_dm = positive_dm.rolling(window=14).sum()
    smooth_negative_dm = negative_dm.rolling(window=14).sum()
    smooth_true_range = true_range.rolling(window=14).sum()
    
    # Calculate +DI and -DI
    positive_di = 100 * (smooth_positive_dm / smooth_true_range)
    negative_di = 100 * (smooth_negative_dm / smooth_true_range)
    
    # Calculate ADMI
    admi = (positive_di - negative_di) / (positive_di + negative_di)
    
    # Multiply Preliminary Factor by ADMI
    intermediate_alpha_factor = preliminary_factor * admi
    
    # Examine the Relationship Between Close Price and Volume
    avg_volume = df['volume'].rolling(window=14).mean()
    close_to_avg_volume = df['close'] / avg_volume
    
    # Assess the Volatility
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=14).std()
    
    # Combine Intermediate Alpha Factor with New Components
    final_alpha_factor = intermediate_alpha_factor * (close_to_avg_volume / volatility)
    
    return final_alpha_factor
