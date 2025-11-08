import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Residual momentum from linear price regression
    x = np.arange(20).reshape(-1, 1)
    def price_residuals(window):
        if len(window) < 20:
            return np.nan
        y = window.values.reshape(-1, 1)
        try:
            coeff = np.linalg.lstsq(x, y, rcond=None)[0][0][0]
            predicted = coeff * 19
            actual = window.iloc[-1]
            return actual - predicted
        except:
            return np.nan
    residual_momentum = close.rolling(window=20).apply(price_residuals, raw=False)
    
    # Volume-confirmed breakout signals
    rolling_high = high.rolling(window=10).max()
    rolling_low = low.rolling(window=10).min()
    volume_ma = volume.rolling(window=10).mean()
    breakout_signal = np.where(close > rolling_high, 1, np.where(close < rolling_low, -1, 0))
    volume_confirmed_breakout = breakout_signal * (volume / volume_ma)
    
    # Liquidity-adjusted volatility regime
    atr = (high - low).rolling(window=14).mean()
    dollar_volume = close * volume
    liquidity_volatility = atr / dollar_volume.rolling(window=14).mean()
    
    heuristics_matrix = residual_momentum + volume_confirmed_breakout - liquidity_volatility
    
    return heuristics_matrix
