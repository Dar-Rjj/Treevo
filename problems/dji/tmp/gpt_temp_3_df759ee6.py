import pandas as pd

def heuristics_v2(df):
    def custom_transform(data):
        # Custom transformation logic
        transformed = (data['close'] / data['open']).apply(lambda x: 0 if x == 0 else math.log(x))
        return transformed

    # Main steps
    step1 = df.apply(custom_transform, axis=1)
    step2 = step1.rolling(window=10).mean()
    step3 = df['high'] - df['low']
    step4 = step3.ewm(span=5, adjust=False).mean()
    heuristics_matrix = 0.6 * step2 + 0.4 * step4
    
    return heuristics_matrix
