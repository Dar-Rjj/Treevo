import pandas as pd
    def aggregate_features(row):
        # Example heuristic: Weighted sum based on simple assumptions about feature importance
        weights = {'open': 0.2, 'high': 0.25, 'low': 0.2, 'close': 0.3, 'volume': 0.05}
        return sum([row[col] * weights[col] for col in df.columns])
    
    heuristics_matrix = df.apply(aggregate_features, axis=1)
    return heuristics_matrix
