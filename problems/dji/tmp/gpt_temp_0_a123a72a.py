import pandas as pd

def heuristics_v2(df):
    def weighted_moving_average(data, window, weight_factor):
        return data.rolling(window=window).apply(lambda x: (x * weight_factor).sum() / weight_factor.sum(), raw=True)

    close = df['close']
    volume = df['volume']
    
    # Calculate weighted moving averages for price and volume
    wma_close_10 = weighted_moving_average(close, 10, range(1, 11))
    wma_close_30 = weighted_moving_average(close, 30, range(1, 31))
    wma_volume_10 = weighted_moving_average(volume, 10, range(1, 11))
    wma_volume_30 = weighted_moving_average(volume, 30, range(1, 31))
    
    # Momentum calculation
    momentum_close = close.pct_change(20)
    
    # Composite heuristic score
    heuristics_matrix = (wma_close_10 - wma_close_30) + (wma_volume_10 - wma_volume_30) + momentum_close
    
    return heuristics_matrix
