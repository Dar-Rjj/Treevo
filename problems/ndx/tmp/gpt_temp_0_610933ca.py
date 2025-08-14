import pandas as pd

def heuristics_v2(df):
    def calculate_momentum(price_series, window=5):
        return price_series.pct_change(window)

    def calculate_mean_reversion(price_series, window=10):
        return (price_series - price_series.rolling(window=window).mean()) / price_series.rolling(window=window).std()

    def calculate_volatility(price_series, window=20):
        return price_series.pct_change().rolling(window=window).std()

    momentum = df['close'].apply(calculate_momentum)
    mean_reversion = df['close'].apply(calculate_mean_reversion)
    volatility = df['close'].apply(calculate_volatility)

    heuristics_matrix = pd.DataFrame({
        'momentum': momentum,
        'mean_reversion': mean_reversion,
        'volatility': volatility
    })

    # Assuming we want the factor to be a simple average of these three for simplicity
    heuristics_factor = heuristics_matrix.mean(axis=1)
    
    return heuristics_matrix
