def heuristics_v2(df):
    def calculate_adx(high, low, close, n):
        tr = pd.DataFrame({'high': high, 'low': low, 'close_prev': close.shift(1)}).apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close_prev']), abs(x['low'] - x['close_prev'])), axis=1)
        atr = tr.rolling(window=n).mean()
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        plus_di = 100 * (plus_dm.rolling(window=n).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=n).sum() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=n).mean()
        
        return adx

    # Calculate ADX over a 14-day period
    adx = calculate_adx(df['high'], df['low'], df['close'], 14)
    
    # Calculate 50-day SMA
    sma_50 = df['close'].rolling(window=50).mean()
    
    # Calculate 20-day SMA
    sma_20 = df['close'].rolling(window=20).mean()
    
    # Combine factors
    heuristics_matrix = adx + (df['close'] - sma_50) * (sma_20 / sma_50)
    
    return heuristics_matrix
