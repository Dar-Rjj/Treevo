import pandas as pd

def heuristics_v2(df):
    def calculate_moving_averages(data, periods=[5, 10, 20]):
        ma = {}
        for p in periods:
            ma[f'ma_{p}'] = data['close'].rolling(window=p).mean()
        return pd.DataFrame(ma, index=data.index)
    
    def calculate_rsi(data, window=14):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        return macd_line - signal_line

    ma_df = calculate_moving_averages(df)
    rsi_series = calculate_rsi(df)
    macd_series = calculate_macd(df)

    heuristics_matrix = pd.concat([ma_df, rsi_series, macd_series], axis=1)
    heuristics_matrix.columns = [f'factor_{i+1}' for i in range(heuristics_matrix.shape[1])]
    return heuristics_matrix
