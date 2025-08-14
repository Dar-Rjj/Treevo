def heuristics_v2(df):
    avg_vol_20 = df['volume'].rolling(window=20).mean()
    avg_vol_200 = df['volume'].rolling(window=200).mean()
    close_change_20 = df['close'].pct_change(periods=20)
    heuristics_matrix = (np.log(avg_vol_20) - np.log(avg_vol_200)) * (np.exp(close_change_20) - 1)
    return heuristics_matrix
