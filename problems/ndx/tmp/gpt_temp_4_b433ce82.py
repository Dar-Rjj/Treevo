import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term Logarithmic Return
    df['short_term_log_return'] = (df['close'] / df['close'].shift(5)).apply(lambda x: 0 if x <= 0 else math.log(x))
    
    # Calculate Long-Term Logarithmic Return
    df['long_term_log_return'] = (df['close'] / df['close'].shift(20)).apply(lambda x: 0 if x <= 0 else math.log(x))

    # Calculate Volume-Weighted Short-Term Logarithmic Return
    df['volume_weighted_short_term_log_return'] = df['short_term_log_return'] * df['volume']
    
    # Calculate Volume-Weighted Long-Term Logarithmic Return
    df['volume_weighted_long_term_log_return'] = df['long_term_log_return'] * df['volume']
    
    # Calculate Short-Term Volatility
    df['short_term_volatility'] = df[['high', 'low']].rolling(window=5).apply(lambda x: (x.max() - x.min()).mean(), raw=False)
    
    # Adjust for Volatility
    df['volatility_adjusted_return'] = (df['volume_weighted_short_term_log_return'] / df['short_term_volatility']) - df['volume_weighted_long_term_log_return']

    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Combine Intraday Volatility and Close-to-Close Return
    df['close_to_close_return'] = (df['close'] / df['close'].shift(1)) - 1
    df['combined_intraday_vol_close_return'] = df['intraday_volatility'] * df['close_to_close_return']
    
    # Calculate Daily Volume Change
    df['daily_volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Aggregate Volume Changes
    df['aggregated_volume_changes'] = df['daily_volume_change'].rolling(window=5).sum()

    # Final Adjustment
    df['alpha_factor'] = (df['volatility_adjusted_return'] + 
                          df['combined_intraday_vol_close_return']) * df['aggregated_volume_changes']

    return df['alpha_factor'].dropna()
