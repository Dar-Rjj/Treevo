import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close Position in Range
    df['close_position_in_range'] = (df['close'] - df['low']) / df['intraday_range']
    
    # Calculate Daily Return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Weighted Daily Return
    df['weighted_daily_return'] = df['daily_return'] * df['volume']
    
    # Combine Initial Factors
    df['combined_initial_factors'] = df['close_position_in_range'] + df['weighted_daily_return']
    
    # Calculate Breakout Strength
    df['breakout_strength'] = df['intraday_range'] / df['intraday_range'].rolling(window=21).sum()
    
    # Calculate Intraday Reversal Score
    df['intraday_reversal_score'] = (df['close'] - df['open']) / df['intraday_range'] * df['volume']
    
    # Combine Breakout and Intraday Factors
    df['combined_breakout_intraday'] = df['combined_initial_factors'] + df['breakout_strength'] + df['intraday_reversal_score']
    
    # Calculate Price Momentum
    df['price_momentum'] = df['close'].pct_change(7) + df['close'].pct_change(21)
    
    # Determine Volume Impact
    df['volume_impact'] = df['volume'] / df['volume'].mean()
    
    # Incorporate Momentum
    df['adjusted_factor'] = df['combined_breakout_intraday'] + df['price_momentum']
    
    # Calculate Volume Activity
    df['volume_activity'] = df['volume'].rolling(window=7).mean() / df['volume'].rolling(window=21).mean()
    
    # Introduce Trend Indicator
    df['trend_indicator'] = df['close'].rolling(window=50).mean() > df['close'].rolling(window=200).mean()
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = (df['combined_breakout_intraday'] + df['price_momentum']) * df['volume_activity'] * df['volume'] * df['close_to_open_return']
    df['final_alpha_factor'] = df['final_alpha_factor'] * df['trend_indicator'].astype(int)
    df['final_alpha_factor'] = df['final_alpha_factor'].replace({0: -df['final_alpha_factor']})
    
    return df['final_alpha_factor']
