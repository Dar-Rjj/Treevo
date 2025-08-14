import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Calculate Intraday Open-Close Return
    df['intraday_open_close_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Short-Term and Long-Term Momentum
    df['short_term_momentum'] = df['close'] - df['close'].rolling(window=10).mean()
    df['long_term_momentum'] = df['close'] - df['close'].rolling(window=20).mean()
    
    # Calculate Volume Trends
    df['short_term_volume_trend'] = df['volume'].rolling(window=10).mean()
    df['long_term_volume_trend'] = df['volume'].rolling(window=20).mean()
    
    # Dynamic Weighting by Momentum and Volume
    df['relative_strength_score'] = (df['short_term_momentum'] > df['long_term_momentum']).astype(int)
    df['volume_ratio'] = df['short_term_volume_trend'] / df['long_term_volume_trend']
    df['volume_ratio_score'] = (df['volume_ratio'] > 1).astype(int)
    df['final_dynamic_score'] = 1 - (df['relative_strength_score'] * df['volume_ratio_score'])
    df['dynamic_weighted_close'] = df['close'] * df['final_dynamic_score']
    
    # Calculate Momentum Indicator
    df['daily_return'] = df['close'].pct_change()
    df['smoothed_returns'] = df['daily_return'].ewm(span=10, adjust=False).mean()
    
    # Calculate Price Deviation
    df['average_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['price_deviation'] = df['open'] - df['average_price']
    
    # Synthesize Combined Factors
    df['combined_intraday_factors'] = (df['intraday_high_low_spread'] * df['volume']) + (df['intraday_open_close_return'] * df['volume'])
    df['weighted_momentum_and_price_deviation'] = (df['smoothed_returns'] * df['volume']) + df['price_deviation']
    df['intermediate_alpha_factor'] = df['combined_intraday_factors'] + df['weighted_momentum_and_price_deviation']
    df['adjusted_by_previous_volume'] = df['intermediate_alpha_factor'].ewm(com=df['volume'].shift(1), adjust=False).mean()
    
    # Integrate Volume Impact
    df['volume_trend'] = df['volume'].pct_change(periods=10)
    df['volume_adjusted_alpha_factor'] = df['adjusted_by_previous_volume'] * df['volume_trend']
    
    # Final Alpha Factor
    df['reversal_factor'] = df['close'] - df['close'].shift(5)
    df['final_alpha_factor'] = df['volume_adjusted_alpha_factor'] * df['dynamic_weighted_close'] - df['reversal_factor']
    
    return df['final_alpha_factor']
