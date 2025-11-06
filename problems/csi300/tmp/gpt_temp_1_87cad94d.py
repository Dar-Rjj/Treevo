def heuristics_v2(df):
    intraday_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
    momentum_accel = (df['close'] - df['open']).rolling(window=5).apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8))
    volume_surge = df['volume'] / df['volume'].rolling(window=20).mean()
    
    volatility_scaled_momentum = momentum_accel / (intraday_range.rolling(window=5).mean() + 1e-8)
    mean_reversion_signal = -((df['close'] - df['close'].rolling(window=10).mean()) / df['close'].rolling(window=10).std()) * volume_surge
    
    heuristics_matrix = volatility_scaled_momentum + mean_reversion_signal
    return heuristics_matrix
