def heuristics_v2(df):
    # Exponential Daily Returns
    df['Daily_Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['Exponential_Daily_Return'] = df['Daily_Return'].ewm(span=5, adjust=False).mean()

    # Short-Term EMA of Daily Returns (5 days)
    df['Short_Term_EMA_Returns'] = df['Exponential_Daily_Return'].ewm(span=5, adjust=False).mean()
    
    # Long-Term EMA of Daily Returns (20 days)
    df['Long_Term_EMA_Returns'] = df['Exponential_Daily_Return'].ewm(span=20, adjust=False).mean()
    
    # Dynamic Difference
    df['Dynamic_Difference'] = df['Short_Term_EMA_Returns'] - df['Long_Term_EMA_Returns']
    
    # Weighted 5-day Moving Average of Daily Returns
    df['Volume_Weighted_Return'] = df['Daily_Return'] * df['volume']
    df['Weighted_5day_MA_Return'] = df['Volume_Weighted_Return'].rolling(window=5).mean()
    
    # Short-Term EMA of Volume (5 days)
    df['Short_Term_EMA_Volume'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Long-Term EMA of Volume (20 days)
    df['Long_Term_EMA_Volume'] = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Volume Momentum
    df['Volume_Momentum'] = df['Short_Term_EMA_Volume'] - df['Long_Term_EMA_Volume']
    
    # 10-day MA of High-Low Difference
    df['High_Low_Diff'] = df['high'] - df['low']
    df['10day_MA_High_Low_Diff'] = df['High_Low_Diff'].rolling(window=10).mean()
    
    # 10-day Volume-Weighted MA of Open-Close Difference
    df['Open_Close_Diff'] = (df['open'] - df['close']) * df['volume']
    df['10day_VW_MA_Open_Close_Diff'] = df['Open_Close_Diff'].rolling(window=10).mean()
    
    # Cumulative Return with Adjustments
    N = 10
    df['Cumulative_Return'] = (df['close'] / df['close'].shift(N)) - 1
    df['N_day_Avg_Volume'] = df['volume'].rolling(window=N).mean()
    df['Volume_Adjusted_Return'] = df['Cumulative_Return'] * df['N_day_Avg_Volume']
    df['Max_High_N_days'] = df['high'].rolling(window=N).max()
    df['Min_Low_N_days'] = df['low'].rolling(window=N).min()
    df['Price_Range_Adjustment'] = df['Max_High_N_days'] - df['Min_Low_N_days']
    df['Adjusted_Cumulative_Return'] = df['Volume_Adjusted_Return'] / df['Price_Range_Adjustment']
    
    # Adjusted High-Low Spread with True Range
    df['Prev_Close'] = df['close'].shift(1)
    df['True_Range'] = df[['high', 'low', 'Prev_Close']].apply(
        lambda x: max(x['high'] - x['low'], x['high'] - x['Prev_Close'], x['Prev_Close'] - x['low']), axis=1
    )
    df['Adjusted_High_Low_Spread'] = df['High_Low_Diff'] + df['True_Range']
    
    # Volume-Weighted Spread with ATR
    df['ATR'] = df['True_Range'].rolling(window=14).mean()
    df['Volume_Weighted_Spread'] = df['volume'] * df['Adjusted_High_Low_Spread']
    df['VWS_ATR'] = df['Volume_Weighted_Spread'] / df['ATR']
    
    # Condition on Close-to-Open Return
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    positive_return_weight = 1.5
