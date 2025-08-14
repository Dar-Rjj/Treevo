def heuristics_v2(df):
    def custom_transform(data):
        transformed = (np.log(data['high']) - np.log(data['low'])) / (np.log(data['open']) - np.log(data['close']) + 1e-6)
        return transformed

    step1 = df.apply(custom_transform, axis=1)
    step2 = step1.rolling(window=20).std()
    step3 = df['volume'].ewm(span=20, adjust=False).mean()
    heuristics_matrix = 0.5 * step2 + 0.5 * step3
    
    return heuristics_matrix
