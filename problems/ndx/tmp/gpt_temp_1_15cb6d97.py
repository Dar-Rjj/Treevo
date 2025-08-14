def heuristics_v2(df):
    def compute_momentum(series, lookback=20):
        return (series / series.shift(lookback)) - 1

    def compute_volatility(series, lookback=20):
        return series.pct_change().rolling(window=lookback).std()

    lookback = 20
    momentum = df.apply(compute_momentum, lookback=lookback)
    volatility = df.apply(compute_volatility, lookback=lookback)
    
    # Placeholder for actual future returns data; assuming it's available in the DataFrame as 'future_returns'
    future_returns = df['future_returns']
    
    weights = df.pct_change().corrwith(future_returns).fillna(0)
    combined_factors = (momentum * weights) + (volatility * (1-weights))
    heuristics_matrix = combined_factors.sum(axis=1)
    
    return heuristics_matrix
