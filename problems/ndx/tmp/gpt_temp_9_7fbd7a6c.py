import numpy as np
def heuristics_v2(df):
    # Calculate Daily Price Change
    df['price_change'] = df['close'].diff()

    # Calculate Weighted Moving Averages
    short_term_weights = np.array([2, 3, 4, 5, 6])
    long_term_weights = np.arange(2, 21)

    def weighted_moving_average(series, weights):
        return (series * weights).sum() / weights.sum()
