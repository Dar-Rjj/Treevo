def heuristics_v2(df):
    # Compute moving averages
    short_window = 10
    long_window = 50
    df['SMA_short'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    
    # Compute RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    rolling_mean = df['close'].rolling(window=20, min_periods=1).mean()
    rolling_std = df['close'].rolling(window=20, min_periods=1).std()
    df['BB_upper'] = rolling_mean + (2 * rolling_std)
    df['BB_lower'] = rolling_mean - (2 * rolling_std)
    
    # Momentum
    df['Momentum'] = df['close'] - df['close'].shift(10)
    
    # Historical Return Correlation (as proxy for weights, simplified in this example, in real setting more complex)
    df['Return'] = df['close'].pct_change().shift(-1)  # Forward return to simulate predictability
    corr_SMA_short = df[['SMA_short', 'Return']].corr().iloc[1, 0]
    corr_SMA_long = df[['SMA_long', 'Return']].corr().iloc[1, 0]
    corr_RSI = df[['RSI', 'Return']].corr().iloc[1, 0]
    corr_BB_upper = df[['BB_upper', 'Return']].corr().iloc[1, 0]
    corr_BB_lower = df[['BB_lower', 'Return']].corr().iloc[1, 0]
    corr_Momentum = df[['Momentum', 'Return']].corr().iloc[1, 0]
    
    # Forming the factor, here we simply use correlations as weights
    df['heuristics_matrix'] = (df['SMA_short']*corr_SMA_short + df['SMA_long']*corr_SMA_long + 
                               df['RSI']*corr_RSI + df['BB_upper']*corr_BB_upper + 
                               df['BB_lower']*corr_BB_lower + df['Momentum']*corr_Momentum)
    
    return heuristics_matrix
