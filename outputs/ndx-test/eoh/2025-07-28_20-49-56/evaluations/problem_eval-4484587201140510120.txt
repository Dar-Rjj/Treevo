def heuristics_v2(df):
    # Calculate the ADX for a 14-day period
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = pd.concat([high - low, 
                    (high - close.shift()).abs(), 
                    (low - close.shift()).abs()], 
                   axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = up_move.where(up_move > down_move, 0).where(up_move > 0, 0)
    minus_dm = down_move.where(down_move > up_move, 0).where(down_move > 0, 0)
    
    plus_di = 100 * (plus_dm.rolling(window=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).sum() / atr)
    
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = 100 * dx.rolling(window=14).mean()
    
    # Calculate the MFI for a 14-day period
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    
    money_ratio = positive_money_flow.rolling(window=14).sum() / negative_money_flow.rolling(window=14).sum()
    mfi = 100 - (100 / (1 + money_ratio))
    
    # Combine ADX and MFI into a single heuristics measure with specific weights
    heuristics_matrix = (adx * 0.6 + mfi * 0.4)
    return heuristics_matrix
