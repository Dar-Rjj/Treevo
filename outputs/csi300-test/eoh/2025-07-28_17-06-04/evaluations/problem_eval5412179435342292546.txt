import numpy as np
    log_vol = np.log(df['volume'])
    avg_log_vol_250 = log_vol.rolling(window=250).mean()
    heuristics_matrix = log_vol / avg_log_vol_250
    return heuristics_matrix
