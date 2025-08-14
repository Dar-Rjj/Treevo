import pandas as pd
    log_volume_price_ratio = df['volume'].apply(np.log) / df['close']
    heuristics_matrix = log_volume_price_ratio.ewm(span=20, adjust=False).mean()
    return heuristics_matrix
