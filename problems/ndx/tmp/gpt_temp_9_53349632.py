import pandas as pd

def heuristics_v2(df):
    def compute_moving_averages(data, window):
        return data.rolling(window=window).mean()
    
    def compute_high_low_ratio(data):
        return data['high'] / data['low']
    
    def compute_log_return(data, window):
        return (data['close'].shift(-window) / data['close']).apply(np.log)
    
    def compute_average_true_range(data, window):
        tr = pd.DataFrame(index=data.index)
        tr['h-l'] = data['high'] - data['low']
        tr['h-pc'] = abs(data['high'] - data['close'].shift(1))
        tr['l-pc'] = abs(data['low'] - data['close'].shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        atr = tr['tr'].rolling(window=window).mean()
        return atr
    
    short_window = 20
    long_window = 100
    log_return_window = 50
    atr_window = 14

    ma_short = compute_moving_averages(df['close'], short_window)
    ma_long = compute_moving_averages(df['close'], long_window)
    hlr_ratio = compute_high_low_ratio(df)
    log_return = compute_log_return(df, log_return_window)
    atr = compute_average_true_range(df, atr_window)
    
    heuristics_matrix = (ma_short - ma_long) * 0.3 + hlr_ratio * 0.4 + log_return * 0.2 + atr * 0.1
    return heuristics_matrix
