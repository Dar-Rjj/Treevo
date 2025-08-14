def heuristics_v2(df):
    def calculate_log_return(series):
        return np.log(series / series.shift(1))
    
    df['log_return'] = calculate_log_return(df['close'])
    
    delta = df['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up = up.rolling(window=14).mean()
    roll_down = down.abs().rolling(window=14).mean()
    
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    heuristics_matrix = df['log_return'] * rsi
    return heuristics_matrix
