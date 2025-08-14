def heuristics_v2(df):
    # Momentum-Based Factors
    def simple_moving_average(price, window):
        return price.rolling(window=window).mean()
