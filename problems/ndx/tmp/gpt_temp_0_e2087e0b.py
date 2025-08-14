def heuristics_v2(df):
    # Calculate Bollinger Bands
    middle_band = df['close'].rolling(window=20).mean()
    std_dev = df['close'].rolling(window=20).std()
    upper_band = middle_band + 2 * std_dev
    lower_band = middle_band - 2 * std_dev
    bb_value = (df['close'] - lower_band) / (upper_band - lower_band)

    # Calculate On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Recent price momentum calculation over the last 5 days
    price_momentum = df['close'].pct_change(periods=5).shift(-5)

    # Combine BB, OBV, and recent price momentum into a single measure
    heuristics_matrix = (bb_value + obv / 1e6 + price_momentum) * df['close'].pct_change().rolling(window=5).mean()

    return heuristics_matrix
