import pandas as pd

def heuristics_v2(df):
    def custom_transform(data):
        # Custom transformation logic
        transformed = (data['close'] - data['open']) / data['volume'].rolling(window=5).mean()
        return transformed

    # Main steps
    step1 = df.apply(custom_transform, axis=1)
    step2 = step1.rolling(window=5).mean()
    step3 = df['high'] - df['low']
    step4 = step3.ewm(span=10, adjust=False).mean()
    heuristics_matrix = 0.7 * step2 + 0.3 * step4
    
    return heuristics_matrix
