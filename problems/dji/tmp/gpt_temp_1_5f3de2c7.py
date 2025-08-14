def heuristics_v2(df):
    def momentum_score(data):
        return (data['close'] - data['close'].shift(7)) / data['close'].shift(7)

    step1 = df.apply(momentum_score, axis=1)
    step2 = df['volume'].ewm(span=20, adjust=False).mean()
    step3 = (df['close'] / df['volume']).rolling(window=10).mean()
    heuristics_matrix = 0.6 * step1 + 0.4 * (step2 * step3)
    
    return heuristics_matrix
