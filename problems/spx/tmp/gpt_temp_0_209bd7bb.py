import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate daily return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Identify trend following momentum with a lookback period of 10 days
    N = 10
    df['trend_momentum'] = (df['close'] - df['close'].shift(N)) / df['close'].shift(N)
    
    # Measure intraday movement
    df['intraday_movement'] = df['high'] - df['low']
    
    # Calculate daily return weighted by volume
    df['vol_weighted_return'] = (df['close'] - df['close'].shift(1)) * df['volume'] / df['close'].shift(1)
    
    # Compute volume change
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Determine high volume days relative to average with a moving average window of 20 days
    M = 20
    df['avg_volume'] = df['volume'].rolling(window=M).mean()
    df['high_volume_day'] = (df['volume'] > df['avg_volume']).astype(int)
    
    # Find trade amount per unit of price
    df['amount_per_price'] = df['amount'] / df['close']
    
    # Analyze trade amount trends with a lookback period of 5 days
    P = 5
    df['amount_trend'] = (df['amount'] - df['amount'].shift(P)) / df['amount'].shift(P)
    
    # Combine trade amount and volume for more complex signals
    df['amount_vol_signal'] = (df['amount'] / df['close']) * (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Evaluate the correlation between price changes and volume with a lag of 1 day
    Q = 1
    df['price_vol_corr'] = df[['close', 'volume']].pct_change().rolling(window=Q+1).corr().unstack().iloc[::2, -1].reset_index(drop=True)
    
    # Investigate the interaction between intraday range and closing price
    df['intraday_close_ratio'] = (df['high'] - df['low']) / df['close']
    
    # Introduce a new factor: Price-Volume Ratio
    df['price_vol_ratio'] = (df['close'] / df['close'].shift(1)) * (df['volume'] / df['volume'].shift(1))
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (
        df['daily_return'] + 
        df['trend_momentum'] + 
        df['intraday_movement'] + 
        df['vol_weighted_return'] + 
        df['volume_change'] + 
        df['high_volume_day'] + 
        df['amount_per_price'] + 
        df['amount_trend'] + 
        df['amount_vol_signal'] + 
        df['price_vol_corr'] + 
        df['intraday_close_ratio'] + 
        df['price_vol_ratio']
    )
    
    # Return the alpha factor as a pandas Series
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('path_to_your_data.csv')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
