def heuristics_v2(df):
    # Calculate the daily price change
    df['price_change'] = df['close'] - df['open']
    
    # Heuristic 1: Price Change over Average Volume (to gauge how much movement per average volume)
    df['heuristic_1'] = df['price_change'] / df['volume']

    # Heuristic 2: Relative Strength Index (RSI) type measure but without normalization for raw momentum indication
    df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
    df['loss'] = df['price_change'].apply(lambda x: -x if x < 0 else 0)
    avg_gain = df['gain'].rolling(window=14).mean()
    avg_loss = df['loss'].rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['heuristic_2'] = 100 - (100 / (1 + rs))

    # Heuristic 3: Cumulative sum of price changes divided by cumulative sum of volumes for observing trend strength
    df['cum_price_change'] = df['price_change'].cumsum()
    df['cum_volume'] = df['volume'].cumsum()
    df['heuristic_3'] = df['cum_price_change'] / df['cum_volume']

    # Constructing the heuristics matrix
    heuristics_matrix = df[['heuristic_1', 'heuristic_2', 'heuristic_3']]
    
    return heuristics_matrix
