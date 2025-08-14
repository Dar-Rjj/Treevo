import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['High'] - df['Low']
    
    # Compute Previous Day's Open-to-Close Return
    df['prev_day_open'] = df['Open'].shift(1)
    df['open_to_close_return'] = df['Close'] - df['prev_day_open']
    
    # Calculate Volume Weighted Average Price (VWAP)
    total_volume = df['Volume']
    vwap_sum = (df['High'] * df['Volume'] + 
                df['Low'] * df['Volume'] + 
                df['Close'] * df['Volume'] + 
                df['Open'] * df['Volume'])
    df['vwap'] = vwap_sum / (4 * total_volume)
    
    # Combine Intraday Momentum and VWAP
    df['combined_value'] = df['vwap'] - df['intraday_high_low_spread']
    df['volume_weighted_combined_value'] = df['combined_value'] * df['Volume']
    
    # Smooth the Factor using Exponential Moving Average (EMA)
    df['alpha_factor'] = df['volume_weighted_combined_value'].ewm(span=5, adjust=False).mean()
    
    return df['alpha_factor']

# Example usage:
# df = pd.DataFrame({
#     'Open': [...],
#     'High': [...],
#     'Low': [...],
#     'Close': [...],
#     'Volume': [...]
# }, index=pd.DatetimeIndex([...]))
# alpha_factor = heuristics_v2(df)
