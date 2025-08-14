def heuristics_v2(df):
    # Momentum Indicators
    def calculate_sma(data, window):
        return data.rolling(window=window).mean()
