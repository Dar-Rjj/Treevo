import numpy as np
def heuristics_v2(df):
    # Trend Following Factors
    df['short_term_wma'] = df['close'].rolling(window=5).apply(lambda x: np.average(x, weights=np.arange(1, 6)), raw=True)
    df['long_term_wma'] = df['close'].rolling(window=20).apply(lambda x: np.average(x, weights=np.arange(1, 21)), raw=True)
    df['wma_crossover'] = df['short_term_wma'] - df['long_term_wma']
    
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['price_minus_ema'] = df['close'] - df['ema_20']
    
    # Momentum Indicators
    def tsi(df, long_period=25, short_period=13):
        diff = df['close'].diff(1)
        abs_diff = diff.abs()
        ema_long = diff.ewm(span=long_period, adjust=False).mean()
        ema_long_abs = abs_diff.ewm(span=long_period, adjust=False).mean()
        ema_short = ema_long.ewm(span=short_period, adjust=False).mean()
        ema_short_abs = ema_long_abs.ewm(span=short_period, adjust=False).mean()
        return (ema_short / ema_short_abs) * 100
