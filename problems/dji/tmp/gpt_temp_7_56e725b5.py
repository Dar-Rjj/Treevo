import numpy as np
    close_prices = df['close']
    log_returns = np.log(close_prices) - np.log(close_prices.shift(1))
    momentum = close_prices.pct_change(5)
    volatility = log_returns.rolling(window=10).std()
    heuristics_matrix = (momentum / volatility).dropna()
    return heuristics_matrix
