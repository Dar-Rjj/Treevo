import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['High'] - df['Low']

    # Compute Previous Day's Close-to-Open Return
    df['prev_close_to_open_return'] = df['Close'].shift(1) - df['Open']

    # Calculate Volume Weighted Average Price (VWAP)
    df['daily_vwap'] = (df['Open'] * df['Volume'] + df['High'] * df['Volume'] + 
                        df['Low'] * df['Volume'] + df['Close'] * df['Volume']) / (4 * df['Volume'])

    # Combine Intraday Momentum and VWAP
    df['combined_momentum'] = df['daily_vwap'] - df['intraday_high_low_spread']
    
    # Weight by Intraday Volume
    df['weighted_combined_momentum'] = df['combined_momentum'] * df['Volume']

    # Smooth the Factor using Exponential Moving Average (EMA) over the last 5 days
    df['smoothed_factor'] = df['weighted_combined_momentum'].ewm(span=5).mean()

    # Calculate Intraday Volume Momentum
    df['intraday_volume_momentum'] = df['Volume'] - df['Volume'].shift(1)

    # Integrate Intraday Volume Momentum with the smoothed factor
    df['final_factor'] = df['smoothed_factor'] + df['intraday_volume_momentum']

    return df['final_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
