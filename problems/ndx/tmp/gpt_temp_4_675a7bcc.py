import pandas as pd

def heuristics_v2(df):
    def simple_moving_average(series, window=10):
        return series.rolling(window=window).mean()

    def exponential_moving_average(series, span=10):
        return series.ewm(span=span, adjust=False).mean()

    def price_rate_of_change(series, n=10):
        return (series / series.shift(n) - 1) * 100

    def volume_rate_of_change(series, n=10):
        return (series / series.shift(n) - 1) * 100
    
    close = df['close']
    sma_10 = simple_moving_average(close)
    ema_10 = exponential_moving_average(close)
    roc_10 = price_rate_of_change(close)
    vroc_10 = volume_rate_of_change(df['volume'])

    # Create a DataFrame to hold the heuristics
    heuristics_matrix = pd.DataFrame({
        'SMA_10': sma_10,
        'EMA_10': ema_10,
        'ROC_10': roc_10,
        'VROC_10': vroc_10
    })

    # Replace NaN values with 0 for simplicity
    heuristics_matrix.fillna(0, inplace=True)

    # Return the heuristics as a single Series object
    return heuristics_matrix
