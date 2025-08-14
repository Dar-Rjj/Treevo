import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df, N=20, M=20, P=20, Q=5, R=20):
    # Calculate daily return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Identify trend following momentum
    df['momentum'] = (df['close'] - df['close'].shift(N)) / df['close'].shift(N)
    
    # Measure intraday movement
    df['intraday_movement'] = df['high'] - df['low']
    
    # Calculate daily return weighted by volume
    df['volume_weighted_return'] = (df['close'] - df['close'].shift(1)) * df['volume'] / df['close'].shift(1)
    
    # Compute volume change
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Determine high volume days relative to average
    df['avg_volume'] = df['volume'].rolling(window=M).mean()
    df['high_volume_day'] = (df['volume'] > df['avg_volume']).astype(int)
    
    # Find trade amount per unit of price
    df['amount_per_price'] = df['amount'] / df['close']
    
    # Analyze trade amount trends
    df['amount_trend'] = (df['amount'] - df['amount'].shift(P)) / df['amount'].shift(P)
    
    # Combine with price and volume for more complex signals
    df['complex_signal'] = (df['amount'] / df['close']) * (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Evaluate the correlation between price changes and volume
    df['price_change'] = df['close'] - df['close'].shift(Q)
    df['volume_change_lag'] = df['volume'] - df['volume'].shift(Q)
    df['corr_price_volume'] = df[['price_change', 'volume_change_lag']].rolling(window=Q).corr().unstack().iloc[::2, 1]
    
    # Investigate the interaction between intraday range and closing price
    df['intraday_range_to_close'] = (df['high'] - df['low']) / df['close']
    
    # Assess the impact of high volume on future returns
    df['future_return'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['high_volume_impact'] = df['high_volume_day'] * df['future_return'].shift(1)
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (
        df['daily_return'] + 
        df['momentum'] + 
        df['intraday_movement'] + 
        df['volume_weighted_return'] + 
        df['volume_change'] + 
        df['high_volume_day'] + 
        df['amount_per_price'] + 
        df['amount_trend'] + 
        df['complex_signal'] + 
        df['corr_price_volume'] + 
        df['intraday_range_to_close'] + 
        df['high_volume_impact']
    )
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df['alpha_factor']
