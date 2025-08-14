def heuristics_v2(df):
    def moving_average_crossover(data, short_window, long_window):
        short_mavg = data.rolling(window=short_window, min_periods=1).mean()
        long_mavg = data.rolling(window=long_window, min_periods=1).mean()
        return (short_mavg > long_mavg).astype(int)

    def momentum(data, window):
        return data - data.shift(window)

    def volume_trend(volume, window):
        return volume.rolling(window=window, min_periods=1).mean().pct_change()

    # Calculate individual components
    ma_crossover = moving_average_crossover(df['close'], 50, 200)
    mom = momentum(df['close'], 14)
    vol_trend = volume_trend(df['volume'], 30)

    # Combine into a single heuristic
    heuristics_matrix = (ma_crossover + mom + vol_trend).sum(axis=1)

    return heuristics_matrix
