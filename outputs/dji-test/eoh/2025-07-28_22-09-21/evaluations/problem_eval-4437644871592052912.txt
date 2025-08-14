def heuristics_v2(df):
    def custom_transform(x):
        return (x - df[x.name].mean()) ** 2 + x.shift(1).fillna(0)
    
    ma_short = df.rolling(window=5).mean()
    ma_long = df.rolling(window=20).mean()
    roc = df.pct_change(periods=10)
    
    heuristics_matrix = (ma_short - ma_long) * roc.apply(custom_transform)
    return heuristics_matrix
