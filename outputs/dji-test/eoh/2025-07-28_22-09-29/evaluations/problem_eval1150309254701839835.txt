def heuristics_v2(df):
    def momentum(df, window):
        return df['close'].pct_change(periods=window).dropna()

    def reversal(df, window):
        return -df['close'].pct_change(periods=window).dropna()
    
    def volume_to_price_ratio(df, window):
        return (df['volume'] / df['close']).rolling(window=window).mean().dropna()

    momentum_10 = momentum(df, 10)
    reversal_5 = reversal(df, 5)
    vol_to_price_10 = volume_to_price_ratio(df, 10)

    heuristics_matrix = pd.DataFrame({
        'momentum_10': momentum_10,
        'reversal_5': reversal_5,
        'vol_to_price_10': vol_to_price_10
    }).fillna(0)  # fill NaN with 0 for simplicity

    return heuristics_matrix
