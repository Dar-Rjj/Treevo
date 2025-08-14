def heuristics_v2(df):
    # Momentum-Based Factors
    def simple_moving_average(df, window):
        return df['close'].rolling(window=window).mean()
