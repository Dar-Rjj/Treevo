import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Movement
    df['daily_price_movement'] = df['close'].diff()
    
    # Calculate Price Change Over Time Window (t-20)
    df['price_change_20'] = df['close'] - df['close'].shift(20)
    
    # Calculate Historical Price Volatility over last 22 days
    df['volatility'] = df['close'].rolling(window=22).std()
    
    # Calculate Momentum (Short-term and Long-term)
    df['momentum_short_term'] = df['close'] / df['close'].shift(7)
    df['momentum_long_term'] = df['close'] / df['close'].shift(25)
    
    # Incorporate Daily Volume Changes
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Aggregate Volume Impact
    def aggregate_volume_impact(series):
        return (series * df['close']).sum() / df['close'].sum()
    df['aggregate_volume_impact'] = df['volume_change'].rolling(window=25).apply(aggregate_volume_impact, raw=False)
    
    # Calculate Volume Direction
    df['volume_direction'] = np.where(df['volume'] > df['volume'].shift(1), 1, -1)
    
    # Combine Price Movement and Volume Direction
    df['combined_movement'] = df['daily_price_movement'] * df['volume_direction']
    
    # Weight by Volume and Inverse Volatility
    df['inverse_volatility'] = 1 / df['volatility']
    df['combined_weights'] = df['volume'] * df['inverse_volatility']
    
    # Final Factor
    df['final_factor'] = df['combined_weights'] * df['price_change_20'] + df['aggregate_volume_impact']
    
    # Compute Exponential Moving Average of the Final Factor
    df['ema_final_factor'] = df['final_factor'].ewm(span=25, adjust=False).mean()
    
    return df['ema_final_factor']
