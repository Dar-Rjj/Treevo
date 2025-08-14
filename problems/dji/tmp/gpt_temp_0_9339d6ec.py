def heuristics_v2(df):
    def calculate_log_return(data):
        log_return = np.log(data['close'] / data['open'])
        return log_return
    
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    step1 = df.apply(calculate_log_return, axis=1)
    step2 = step1.rolling(window=5).mean()
    step3 = df['close'].apply(calculate_rsi, axis=1)
    step4 = step3.ewm(span=30, adjust=False).mean()
    heuristics_matrix = 0.6 * step2 + 0.4 * step4
    
    return heuristics_matrix
