import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range and Cumulative Moves
    df['intraday_move'] = df['high'] - df['low']
    
    # 10-Day Cumulative Range
    df['daily_range'] = df['high'] - df['low']
    df['cumulative_range_10'] = df['daily_range'].rolling(window=10).sum()
    
    # Measure Intraday Deviation
    df['intraday_mean_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Smooth the Intraday and Cumulative Trends
    df['ema_intraday_move'] = df['intraday_move'].ewm(span=21, adjust=False).mean()
    df['ema_cumulative_range_10'] = df['cumulative_range_10'].ewm(span=21, adjust=False).mean()
    
    # Calculate Price Momentum
    df['close_trend'] = df['close'].pct_change(21)
    df['ema_close_trend'] = df['close_trend'].ewm(span=21, adjust=False).mean()
    df['momentum_score'] = df['ema_close_trend'].diff()
    
    # Volume Confirmation
    df['volume_trend'] = df['volume'].pct_change(21)
    df['ema_volume_trend'] = df['volume_trend'].ewm(span=21, adjust=False).mean()
    df['volume_score'] = df['ema_volume_trend'].diff()
    
    # Weight by Volume and Amount
    df['adj_volume'] = df['amount'] / df['close']
    
    # Incorporate Trend Following Signal
    df['ema_close_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_close_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['momentum_signal'] = df['ema_close_12'] - df['ema_close_26']
    
    # Incorporate High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    df['avg_close'] = df['close'].rolling(window=21).mean()
    df['scaled_high_low_spread'] = df['high_low_spread'] / df['avg_close']
    
    # Intraday Momentum Reversal
    df['prev_day_close_open_return'] = df['close'].shift(1) - df['open'].shift(1)
    df['intraday_momentum_reversal'] = df['prev_day_close_open_return'] - (df['high_low_spread'] * df['volume'])
    
    # Combine Components into Alpha Factor
    df['alpha_factor'] = (
        df['intraday_deviation'] * 
        df['momentum_score'] * 
        df['ema_cumulative_range_10'] * 
        df['adj_volume'] * 
        df['volume_score'] * 
        df['momentum_signal'] + 
        df['scaled_high_low_spread'] + 
        df['intraday_momentum_reversal']
    )
    
    return df['alpha_factor']
