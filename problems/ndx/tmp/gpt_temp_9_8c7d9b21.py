def heuristics_v2(df):
    # Accumulation/Distribution Oscillator (ADOSC)
    adosc = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']
    adosc_ema = adosc.ewm(span=20, adjust=False).mean()
    
    # 14-day Simple Moving Average (SMA)
    sma_14 = df['close'].rolling(window=14).mean()
    
    # Logarithmic difference between closing price and SMA
    close_to_sma_log_diff = np.log(df['close'] - sma_14)
    
    # Composite heuristic matrix
    heuristics_matrix = 0.6 * adosc_ema + 0.4 * close_to_sma_log_diff
    
    return heuristics_matrix
