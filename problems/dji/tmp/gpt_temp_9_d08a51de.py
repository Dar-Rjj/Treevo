import pandas as pd
import pandas as pd

def heuristics(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Delta
    df['close_to_open_delta'] = df['close'] - df['open']
    
    # Volume-Weighted Intraday Reversal Score
    df['reversal_score'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['vol_weighted_reversal'] = df['reversal_score'] * df['volume']
    
    # Intraday Momentum Adjusted Reversal Score
    df['price_momentum'] = df['close'] - df['close'].shift(1)
    df['adjusted_reversal'] = df['reversal_score'] * (df['volume'] * df['price_momentum'])
    df['adjusted_reversal'] += df['close_to_open_delta'] if df['close_to_open_delta'] > 0 else -df['close_to_open_delta']
    
    # Identify High-Low Range Ratio
    df['daily_high_low_range'] = df['high'] - df['low']
    df['avg_high_low_range'] = df['daily_high_low_range'].rolling(window=21).sum()
    df['avg_volume'] = df['volume'].rolling(window=21).sum()
    
    # Calculate Breakout Strength
    df['breakout_strength'] = df['daily_high_low_range'] / df['avg_high_low_range']
    
    # Calculate Volume Weighted Momentum
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['vol_weighted_momentum'] = df['price_change'] * df['volume']
    
    # Combine Breakout Strength with Volume Weighted Momentum
    df['combined_breakout_momentum'] = df['breakout_strength'] * df['vol_weighted_momentum']
    
    # Calculate Price Momentum
    df['7_day_return'] = df['close'].pct_change(periods=7)
    df['21_day_return'] = df['close'].pct_change(periods=21)
    df['price_momentum'] = df['7_day_return'] + df['21_day_return']
    
    # Calculate Volume Activity
    df['7_day_avg_vol'] = df['volume'].rolling(window=7).mean()
    df['21_day_avg_vol'] = df['volume'].rolling(window=21).mean()
    df['volume_activity'] = df['7_day_avg_vol'] / df['21_day_avg_vol']
    
    # Introduce Trend Indicator
    df['50_day_ma'] = df['close'].rolling(window=50).mean()
    df['200_day_ma'] = df['close'].rolling(window=200).mean()
    df['trend_indicator'] = df['50_day_ma'] > df['200_day_ma']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['combined_breakout_momentum'] * df['price_momentum'] * df['volume_activity']
    df['final_alpha_factor'] *= df['trend_indicator'].astype(int) * 2 - 1
    
    # Incorporate Intraday Volatility
    df['10_day_ema_intraday_range'] = df['intraday_range'].ewm(span=10, adjust=False).mean()
    df['20_day_sma_intraday_range'] = df['intraday_range'].rolling(window=20).mean()
    df['final_alpha_factor'] *= 1.5 if df['10_day_ema_intraday_range'] > df['20_day_sma_intraday_range'] else 1.0
    
    return df['final_alpha_factor']
