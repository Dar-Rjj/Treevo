def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume'].sum()
    df['total_dollar_value'] = (df['volume'] * df['close']).sum()
    df['vwap'] = df['total_dollar_value'] / df['total_volume']
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Adaptive Short-Term Momentum
    df['5_day_volatility'] = df['close'].rolling(window=5).std()
    df['short_term_momentum_period'] = df['5_day_volatility'].apply(lambda x: 5 if x > df['5_day_volatility'].mean() else 10)
    df['short_term_momentum'] = df['vwap_deviation'].rolling(window=df['short_term_momentum_period']).sum()
    df['cumulative_vwap_deviation'] += df['short_term_momentum']
    
    # Integrate Adaptive Medium-Term Momentum
    df['10_day_volatility'] = df['close'].rolling(window=10).std()
    df['medium_term_momentum_period'] = df['10_day_volatility'].apply(lambda x: 10 if x > df['10_day_volatility'].mean() else 20)
    df['medium_term_momentum'] = df['vwap_deviation'].rolling(window=df['medium_term_momentum_period']).sum()
    df['cumulative_vwap_deviation'] += df['medium_term_momentum']
    
    # Integrate Adaptive Long-Term Momentum
    df['20_day_volatility'] = df['close'].rolling(window=20).std()
    df['long_term_momentum_period'] = df['20_day_volatility'].apply(lambda x: 20 if x > df['20_day_volatility'].mean() else 30)
    df['long_term_momentum'] = df['vwap_deviation'].rolling(window=df['long_term_momentum_period']).sum()
    df['cumulative_vwap_deviation'] += df['long_term_momentum']
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['absolute_vwap_deviation'] = (df['close'] - df['vwap']).abs()
    df['intraday_volatility'] = df['high_low_range'] + df['absolute_vwap_deviation']
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['cumulative_vwap_deviation'] 
                          + df['short_term_momentum'] 
                          + df['medium_term_momentum'] 
                          + df['long_term_momentum'] 
                          + df['intraday_volatility'])
    
    return df['alpha_factor']
