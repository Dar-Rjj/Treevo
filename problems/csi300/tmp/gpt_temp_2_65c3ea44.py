def heuristics_v2(df):
    # Calculate VWAP
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # VWAP Difference from Open and Close
    df['VWAP_Diff_Open'] = df['VWAP'] - df['Open']
    df['VWAP_Diff_Close'] = df['VWAP'] - df['Close']
    
    # Calculate Short-Term Momentum
    df['10_day_Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Calculate Long-Term Momentum
    df['60_day_Momentum'] = df['Close'] - df['Close'].shift(60)
    
    # Calculate Daily Returns
    df['Daily_Returns'] = df['Close'].diff()
    
    # Calculate Short-Term Volatility
    df['10_day_Volatility'] = df['Daily_Returns'].rolling(window=10).std()
    
    # Calculate Long-Term Volatility
    df['60_day_Volatility'] = df['Daily_Returns'].rolling(window=60).std()
    
    # Combine Momentum and Volatility
    df['Short_Term_Momentum_Vol_Product'] = df['10_day_Momentum'] * df['10_day_Volatility']
    df['Long_Term_Momentum_Vol_Product'] = df['60_day_Momentum'] * df['60_day_Volatility']

    # Final Alpha Factor
    df['VWAP_Adjusted_Short_Term_Momentum'] = df['VWAP_Diff_Open'] * df['10_day_Momentum']
    df['VWAP_Adjusted_Long_Term_Momentum'] = df['VWAP_Diff_Close'] * df['60_day_Momentum']
    alpha_factor = (
        df['VWAP_Adjusted_Short_Term_Momentum'] 
        - df['VWAP_Adjusted_Long_Term_Momentum'] 
        + df['Short_Term_Momentum_Vol_Product'] 
        - df['Long_Term_Momentum_Vol_Product']
    )
    
    return alpha_factor
