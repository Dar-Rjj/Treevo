def heuristics_v2(df):
    def custom_heuristic(sub_df):
        high_low_avg = (sub_df['high'].mean() + sub_df['low'].mean()) / 2
        return high_low_avg - sub_df['close'][-1]
    
    heuristics_matrix = df.rolling(window=10).apply(custom_heuristic, raw=False)
    return heuristics_matrix
