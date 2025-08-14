import pandas as pd
import pandas as pd

def heuristics_v2(df, lookback_period=20, moving_avg_window=10, amount_lookback=5, lag=5):
    # Calculate daily return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Identify trend following momentum
    df['momentum'] = (df['close'] - df['close'].shift(lookback_period)) / df['close'].shift(lookback_period)
    
    # Measure intraday movement
    df['intraday_movement'] = (df['high'] - df['low'])
    
    # Evaluate the impact of overnight gaps
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate daily return weighted by volume
    df['weighted_daily_return'] = (df['close'] - df['close'].shift(1)) * df['volume'] / df['close'].shift(1)
    
    # Compute volume change
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Analyze the ratio of current volume to average volume
    df['avg_volume'] = df['volume'].rolling(window=moving_avg_window).mean()
    df['volume_ratio'] = df['volume'] / df['avg_volume']
    
    # Find trade amount per unit of price
    df['amount_per_price'] = df['amount'] / df['close']
    
    # Analyze trade amount trends
    df['amount_trend'] = (df['amount'] - df['amount'].shift(amount_lookback)) / df['amount'].shift(amount_lookback)
    
    # Investigate the relationship between trade amount and intraday range
    df['amount_range'] = (df['amount'] / df['close']) * (df['high'] - df['low'])
    
    # Evaluate the correlation between price changes and volume
    df['price_change'] = df['close'] - df['close'].shift(lag)
    df['volume_change_lagged'] = df['volume'] - df['volume'].shift(lag)
    df['correlation'] = df[['price_change', 'volume_change_lagged']].rolling(window=lag).corr().iloc[::2, 1]
    
    # Investigate the interaction between intraday range and closing price
    df['range_to_close'] = (df['high'] - df['low']) / df['close']
    
    # Analyze the joint effect of daily return and volume change
    df['joint_effect'] = df['daily_return'] * df['volume_change']
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['daily_return'] + df['momentum'] + df['intraday_movement'] + 
                          df['overnight_gap'] + df['weighted_daily_return'] + df['volume_change'] + 
                          df['volume_ratio'] + df['amount_per_price'] + df['amount_trend'] + 
                          df['amount_range'] + df['correlation'] + df['range_to_close'] + df['joint_effect'])
    
    return df['alpha_factor']
