def heuristics_v2(df):
    # Adaptive Simple Moving Average (ASMA)
    def adaptive_sma(close, period, volatility_factor):
        rolling_std = close.rolling(window=period).std()
        adjusted_period = period * (1 + volatility_factor * rolling_std)
        return close.rolling(window=adjusted_period.astype(int)).mean()
