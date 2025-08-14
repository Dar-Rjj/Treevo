import pandas as pd

def heuristics_v2(df):
    def custom_transform(data):
        avg_volume = data['volume'].rolling(window=10).mean()
        transformed = (data['close'] - data['open']) / avg_volume
        return transformed
    
    step1 = df.apply(custom_transform, axis=1)
    step2 = step1.rolling(window=5).mean()
    step3 = df['high'] - df['low']
    step4 = step3.ewm(span=20, adjust=False).mean()
    heuristics_matrix = 0.6 * step2 + 0.4 * step4
    
    return heuristics_matrix
