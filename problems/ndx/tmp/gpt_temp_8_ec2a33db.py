import pandas as pd
    # Calculate the average of high and low for each day
    avg_high_low = (df['high'] + df['low']) / 2
    # Compute the rolling mean of the average high-low, using a 5-day window as an example
    rolling_avg_high_low = avg_high_low.rolling(window=5).mean()
    # Calculate the difference between the close price and the rolling mean of the high-low average
    price_diff = df['close'] - rolling_avg_high_low
    # Calculate the daily change in volume
    volume_change = df['volume'].pct_change()
    # The heuristic is the product of the price difference and the volume change, shifted by one to avoid look-ahead bias
    heuristics_matrix = price_diff * volume_change.shift(1)
    return heuristics_matrix
