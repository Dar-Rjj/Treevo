import pandas as pd

def heuristics_v2(df):
    def alpha1(data):
        return data['close'] - data['open']
    
    def alpha2(data):
        return (data['high'] - data['low']) / data['volume']
    
    def alpha3(data):
        return (data['close'] - data['open']) * (data['volume'] / data['low'])
    
    heuristics_matrix = pd.DataFrame({
        'alpha1': alpha1(df),
        'alpha2': alpha2(df),
        'alpha3': alpha3(df)
    })
    
    return heuristics_matrix
