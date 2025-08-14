import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Intraday Open-Close Return
    intraday_open_close_return = (df['close'] - df['open']) / df['open']
    
    # Compute Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / df['close']
    
    # Calculate Short-Term Momentum
    short_term_momentum = df['close'] - df['close'].rolling(window=10).mean()
    
    # Calculate Long-Term Momentum
    long_term_momentum = df['close'] - df['close'].rolling(window=20).mean()
    
    # Determine Relative Strength
    relative_strength_score = (short_term_momentum > long_term_momentum).astype(int)
    
    # Calculate Volume Trend
    short_term_volume_trend = df['volume'].rolling(window=10).mean()
    long_term_volume_trend = df['volume'].rolling(window=20).mean()
    
    # Determine Volume Ratio
    volume_ratio_score = (short_term_volume_trend / long_term_volume_trend).apply(lambda x: 1 if x > 1 else 0)
    
    # Combine Scores
    dynamic_score = 1 - (relative_strength_score * volume_ratio_score)
    
    # Adjust Intraday High-Low Spread
    adjusted_intraday_high_low_spread = intraday_high_low_spread - df['open']
    
    # Smooth Absolute Movement
    smoothed_absolute_movement = adjusted_intraday_high_low_spread.abs().rolling(window=10).mean()
    
    # Compute Price Deviation
    average_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    price_deviation = df['close'] - average_price
    
    # Combine Intraday Factors
    combined_intraday_factors = smoothed_absolute_movement + intraday_open_close_return * df['volume']
    
    # Combine Momentum and Price Deviation
    weighted_momentum = short_term_momentum * df['volume']
    combined_momentum_and_price_deviation = weighted_momentum + price_deviation
    
    # Adjust by Intraday Range Momentum
    current_day_intraday_range = df['high'] - df['low']
    previous_day_intraday_range = (df['high'].shift(1) - df['low'].shift(1)).fillna(current_day_intraday_range)
    intraday_range_momentum = current_day_intraday_range - previous_day_intraday_range
    
    # Compute Open Price Trend
    open_price_trend = df['open'].rolling(window=10).apply(lambda x: pd.Series(x).index[-1] - pd.Series(x).index[0], raw=False)
    
    # Volume Confirmation
    volume_change = df['volume'] - df['volume'].shift(1).fillna(df['volume'])
    
    # Final Alpha Factor
    final_alpha_factor = (smoothed_absolute_movement * open_price_trend + 
                          combined_intraday_factors + 
                          combined_momentum_and_price_deviation + 
                          intraday_range_momentum) * volume_change * dynamic_score
    
    return final_alpha_factor
