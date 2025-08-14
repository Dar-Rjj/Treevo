import pandas as pd

def heuristics_v2(df):
    price_range = (df['high'] - df['low']).apply(lambda x: max(1, x))  # Ensure log is defined
    log_price_volume_adj = (price_range * df['volume']).apply(lambda x: pd.np.log(x))
    momentum = df['close'].pct_change(7).rolling(window=7).mean()
    heuristics_matrix = 0.6 * log_price_volume_adj + 0.4 * momentum
    return heuristics_matrix
