def heuristics_v2(df):
    # Momentum Indicators
    def simple_moving_average(series, window):
        return series.rolling(window=window).mean()
