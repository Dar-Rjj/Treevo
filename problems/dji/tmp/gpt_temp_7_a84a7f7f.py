def heuristics_v2(df):
    def calculate_factor(data):
        close_prices = data['close']
        volumes = data['volume']
        
        # Calculate 5-day and 20-day moving averages of closing prices
        ma_5 = close_prices.rolling(window=5).mean()
        ma_20 = close_prices.rolling(window=20).mean()
        
        # Volume adjustment: today's volume / 20-day average volume
        vol_adj = volumes / volumes.rolling(window=20).mean()
        
        # Alpha factor: (MA5 - MA20) * Volume Adjustment
        factor = (ma_5 - ma_20) * vol_adj
        
        return factor
    
    heuristics_matrix = df.apply(calculate_factor, axis=1)
    return heuristics_matrix
