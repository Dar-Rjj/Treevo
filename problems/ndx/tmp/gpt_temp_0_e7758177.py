def heuristics_v2(df):
    # Calculate Price Momentum with Volume Adjustment
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['Price_Diff'] = df['close'] - df['SMA_5']
    df['Momentum_Score'] = df['Price_Diff'] / df['SMA_5']
    df['Cum_Volume_5'] = df['volume'].rolling(window=5).sum()
    df['Adjusted_Momentum_Score'] = df['Momentum_Score'] * df['Cum_Volume_5']
    
    # Calculate High-to-Low Price Range
    df['Range'] = df['high'] - df['low']
    
    # Calculate Trading Intensity
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Amount_Change'] = df['amount'] - df['amount'].shift(1)
    df['Trading_Intensity'] = df['Volume_Change'] / df['Amount_Change']
    
    # Weight the Range by Trading Intensity
    trading_intensity_scaled = df['Trading_Intensity'] * 1000
    df['Weighted_Range'] = trading_intensity_scaled * df['Range']
    
    # Combine Momentum and Weighted Range
    df['Combined_Momentum_Range'] = df['Adjusted_Momentum_Score'] + df['Weighted_Range']
    
    # Calculate Daily Price Change
    df['Daily_Price_Change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Weighted Moving Averages
    def wma(data, weights):
        return (data * weights).sum() / weights.sum()
