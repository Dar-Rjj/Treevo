import pandas as pd

def heuristics_v2(df):
    step1 = df['close'] - df['close'].shift(10)
    step2 = (df['volume'] / (df['high'] - df['low'] + 1)).apply(lambda x: max(1e-6, x))
    step3 = step2.apply(np.log).ewm(span=20, adjust=False).mean()
    heuristics_matrix = 0.6 * step1 + 0.4 * step3
    return heuristics_matrix
