import pandas as pd
    short_window = 5
    medium_window = 20
    long_window = 60
    very_long_window = 120
    volume_weight = 0.2
    close_price_weight = 0.4
    price_change_weight = 0.4

    # Calculate moving averages
    short_ma = df['close'].rolling(window=short_window).mean()
    medium_ma = df['close'].rolling(window=medium_window).mean()
    long_ma = df['close'].rolling(window=long_window).mean()
    very_long_ma = df['close'].rolling(window=very_long_window).mean()

    # Calculate volume average
    volume_avg = df['volume'].rolling(window=medium_window).mean()

    # Calculate price change
    price_change = (df['close'] - df['open']) / df['open']

    # Heuristic score calculation
    heuristics_score = (close_price_weight * (short_ma - very_long_ma) +
                        price_change_weight * price_change.rolling(window=medium_window).mean() +
                        volume_weight * (volume_avg - df['volume'].rolling(window=long_window).mean()))

    return heuristics_matrix
