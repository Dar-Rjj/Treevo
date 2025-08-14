import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume Moving Average
    df['volume_moving_avg'] = df['volume'].rolling(window=20).mean()
    
    # Determine if there is a volume shock
    df['volume_shock'] = df['volume'] > df['volume_moving_avg']
    
    # Compute the weighted combination
    df['recent_weight_intraday'] = 0.7
    df['recent_weight_close_open'] = 0.3
    df['older_weight_intraday'] = 0.5
    df['older_weight_close_open'] = 0.5
    
    # Adjust weights based on volume shock
    df['final_weight_intraday'] = np.where(df['volume_shock'], 0.8, df['recent_weight_intraday'])
    df['final_weight_close_open'] = np.where(df['volume_shoot'], 0.2, df['recent_weight_close_open'])
    
    # Combine the factors using the final weights
    df['intraday_volatility_adjusted_return'] = (
        df['final_weight_intraday'] * df['intraday_range'] + 
        df['final_weight_close_open'] * df['close_to_open_return']
    )
    
    return df['intraday_volatility_adjusted_return']

# Example usage
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor = heuristics_v2(df)
# print(factor)
