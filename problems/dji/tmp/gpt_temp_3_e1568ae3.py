import numpy as np
def heuristics_v2(df):
    # Calculate Daily High-Low Range Percentage
    df['daily_range_pct'] = (df['high'] - df['low']) / df['open']
    
    # Calculate Volume-Weighted Intraday Momentum
    df['volume_weighted_momentum'] = (df['close'] - df['open']) * (df['volume'] / df['volume'].rolling(window=20).max())
    
    # Combine Intraday High-Low Range and Volume-Weighted Intraday Momentum
    df['intraday_combined'] = df['daily_range_pct'] * df['volume_weighted_momentum']
    
    # Calculate Momentum Difference
    df['momentum_diff'] = (df['high'] - df['low']) - df['high'].shift(1) + df['low'].shift(1)
    
    # Calculate VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_diff'] = df['vwap'] - df['vwap'].shift(1)
    
    # Calculate Open-Close Spread Difference
    df['open_close_spread_diff'] = (df['close'] - df['open']) - (df['close'].shift(1) - df['open'].shift(1))
    
    # Final Combined Indicator
    df['final_combined'] = df['momentum_diff'] + df['vwap_diff'] + df['open_close_spread_diff']
    
    # Adjust for Price Trend
    df['short_term_trend'] = df['close'].rolling(window=5).mean()
    df['long_term_trend'] = df['close'].rolling(window=20).mean()
    df['trend_adjustment'] = df['long_term_trend'] - df['short_term_trend']
    df['adjusted_combined'] = df['final_combined'] * df['trend_adjustment']
    
    # Adjust for Volatility
    df['rolling_volatility'] = (df['high'] - df['low']).rolling(window=20).std()
    df['volatility_adjusted_combined'] = df['adjusted_combined'] / df['rolling_volatility']
    
    # Calculate Cumulative Log Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
