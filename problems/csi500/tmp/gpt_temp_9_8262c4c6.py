import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Price oscillation mean reversion
    price_center = (high.rolling(10).max() + low.rolling(10).min()) / 2
    oscillation_signal = (price_center - close) / (high.rolling(10).max() - low.rolling(10).min())
    
    # Liquidity-adjusted volatility scaling
    dollar_volume = amount.rolling(5).mean()
    vol_std = close.pct_change().rolling(10).std()
    liquidity_factor = np.log1p(dollar_volume) * vol_std
    scaled_volatility = close.pct_change(3) / (liquidity_factor + 1e-8)
    
    # Cross-sectional momentum breakout
    close_rank = close.rolling(5).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min()))
    volume_trend = volume.rolling(5).apply(lambda x: np.polyfit(range(5), x, 1)[0])
    momentum_breakout = close_rank * np.sign(volume_trend)
    
    # Combine factors
    heuristics_matrix = (oscillation_signal.rank() + 
                        scaled_volatility.rank() * 0.6 + 
                        momentum_breakout.rank() * 0.8)
    
    return heuristics_matrix
