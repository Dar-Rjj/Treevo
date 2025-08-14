def heuristics_v2(df):
    def calculate_alpha_factor(data):
        data['momentum'] = data['close'] - data['close'].rolling(window=20).mean()
        data['heuristic'] = data['momentum'].rolling(window=10).mean()
        return data['heuristic']
    
    heuristics_matrix = df.groupby(level='date').apply(calculate_alpha_factor)
    return heuristics_matrix
