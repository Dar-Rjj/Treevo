def heuristics_v2(df):
    def calculate_average_true_range(df, window):
        true_range = df[['high', 'low']].diff(axis=1).iloc[:, 0].abs()
        true_range = pd.concat([true_range, (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    def calculate_highest_high(df, window):
        return df['high'].rolling(window=window).max()

    # Calculate factors
    average_true_range = calculate_average_true_range(df, 10)
    highest_high = calculate_highest_high(df, 20)
    close_to_highest_high_ratio = df['close'] / highest_high

    # Combine factors
    heuristics_matrix = 0.5 * average_true_range + 0.5 * close_to_highest_high_ratio
    
    return heuristics_matrix
