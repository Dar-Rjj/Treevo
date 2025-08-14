import pandas as pd

def heuristics_v2(df):
    def compute_rsi(data, window):
        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(window=window).mean()
        roll_down = down.abs().rolling(window=window).mean()
        rs = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + rs))
    
    def compute_adx(data, period):
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr = pd.Series([high - low, 
                        abs(high - close.shift(1)), 
                        abs(low - close.shift(1))]).max(axis=0)
        atr = tr.rolling(window=period).mean()
        
        pos_dm = high.diff(1)
        neg_dm = low.diff(1).abs()
        pos_dm[pos_dm <= 0] = 0
        neg_dm[neg_dm <= 0] = 0
        pos_dm[neg_dm > pos_dm] = 0
        neg_dm[pos_dm > neg_dm] = 0
        
        pos_di = 100 * (pos_dm.rolling(window=period).sum() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).sum() / atr)
        
        dx = (abs(pos_di - neg_di) / (pos_di + neg_di)).abs() * 100
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def compute_price_volume_ratio(data):
        return data['close'] / data['volume']
    
    rsi_window = 14
    adx_period = 14
    
    rsi = compute_rsi(df['close'], rsi_window)
    adx = compute_adx(df, adx_period)
    price_volume_ratio = compute_price_volume_ratio(df)
    
    heuristics_matrix = rsi * 0.3 + adx * 0.3 + price_volume_ratio * 0.4
    return heuristics_matrix
