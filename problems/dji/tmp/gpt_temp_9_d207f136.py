import numpy as np
    geom_mean_price = np.exp((np.log(df['high']) + np.log(df['low']) + np.log(df['close'])) / 3)
    volume_geom_mean_ratio = df['volume'] / geom_mean_price
    heuristics_matrix = volume_geom_mean_ratio.ewm(span=14, adjust=False).mean()
    return heuristics_matrix
