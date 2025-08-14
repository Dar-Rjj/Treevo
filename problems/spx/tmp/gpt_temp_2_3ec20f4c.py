import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate price momentum
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['weekly_return'] = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
    df['monthly_return'] = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    
    # Analyze price volatility
    df['daily_log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['daily_log_return'].rolling(window=21).std()
    
    # Investigate trading volume
    df['daily_volume_change'] = df['volume'] - df['volume'].shift(1)
    df['weekly_volume_change'] = df['volume'] - df['volume'].shift(7)
    df['monthly_volume_change'] = df['volume'] - df['volume'].shift(30)
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=21).mean()
    
    # Examine trading amount
    df['daily_amount_change'] = df['amount'] - df['amount'].shift(1)
    df['weekly_amount_change'] = df['amount'] - df['amount'].shift(7)
    df['monthly_amount_change'] = df['amount'] - df['amount'].shift(30)
    df['amount_ratio'] = df['amount'] / df['amount'].rolling(window=21).mean()
    
    # Combine price and volume information
    df['price_volume_corr'] = df['daily_return'].rolling(window=21).corr(df['daily_volume_change'])
    df['daily_money_flow'] = (df['close'] - df['close'].shift(1)) * df['volume']
    df['cumulative_money_flow'] = df['daily_money_flow'].rolling(window=21).sum()
    
    # Analyze high and low prices
    df['high_low_spread'] = df['high'] - df['low']
    df['true_range'] = df[['high' - df['low'], 
                           abs(df['high'] - df['close'].shift(1)), 
                           abs(df['low'] - df['close'].shift(1))]].max(axis=1)
    
    # Identify trading signals
    df['days_since_52_week_high'] = (df['high'].rolling(window=52*5).max() == df['high']).astype(int).cumsum()
    df['days_since_52_week_low'] = (df['low'].rolling(window=52*5).min() == df['low']).astype(int).cumsum()
    
    df['close_above_open_percent'] = (df['close'] > df['open']).rolling(window=21).mean() * 100
    df['close_below_open_percent'] = (df['close'] < df['open']).rolling(window=21).mean() * 100
    
    # Combine all factors into a single alpha factor
    alpha_factor = (
        df['daily_return'] + 
        df['weekly_return'] + 
        df['monthly_return'] + 
        df['volatility'] + 
        df['daily_volume_change'] + 
        df['weekly_volume_change'] + 
        df['monthly_volume_change'] + 
        df['volume_ratio'] + 
        df['daily_amount_change'] + 
        df['weekly_amount_change'] + 
        df['monthly_amount_change'] + 
        df['amount_ratio'] + 
        df['price_volume_corr'] + 
        df['cumulative_money_flow'] + 
        df['high_low_spread'] + 
        df['true_range'] + 
        df['days_since_52_week_high'] + 
        df['days_since_52_week_low'] + 
        df['close_above_open_percent'] + 
        df['close_below_open_percent']
    )
    
    return alpha_factor
