import pandas as pd

def heuristics_v2(df):
    def compute_rsi(data, window=14):
        delta = data.diff(1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        rma_up = up.ewm(alpha=1/window, adjust=False).mean()
        rma_down = down.ewm(alpha=1/window, adjust=False).mean().abs()
        rs = rma_up / rma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def compute_atr(data, window=14):
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = ranges.rolling(window=window).mean()
        return atr
    
    def compute_log_diff_high_low(data):
        log_diff = (data['high'] - data['low']).apply(lambda x: x if x == 0 else math.log(x))
        return log_diff
    
    rsi = compute_rsi(df['close'])
    atr = compute_atr(df)
    log_diff_high_low = compute_log_diff_high_low(df)
    
    heuristics_matrix = (rsi * 0.5) + (atr * 0.3) + (log_diff_high_low * 0.2)
    return heuristics_matrix
