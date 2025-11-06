import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price acceleration momentum (3-day ROC of 5-day ROC)
    roc_5 = close.pct_change(5)
    price_accel = roc_5.pct_change(3)
    
    # Volatility-adjusted mean reversion
    atr = (high - low).rolling(14).mean()
    price_position = (close - close.rolling(21).mean()) / atr
    
    # Volume-confirmed reversal signals
    volume_rank = volume.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volume_spike = volume / volume.rolling(20).mean() - 1
    
    # Dynamic threshold crossover
    upper_threshold = price_accel.rolling(10).quantile(0.7)
    lower_threshold = price_accel.rolling(10).quantile(0.3)
    
    # Signal generation
    momentum_signal = np.where(price_accel > upper_threshold, 1, 
                              np.where(price_accel < lower_threshold, -1, 0))
    
    reversal_signal = np.where((price_position < -1) & (volume_rank > 0.7), 1,
                              np.where((price_position > 1) & (volume_rank > 0.7), -1, 0))
    
    # Factor combination with volume confirmation
    heuristics_matrix = momentum_signal * 0.6 + reversal_signal * 0.4
    heuristics_matrix = pd.Series(heuristics_matrix, index=df.index)
    
    return heuristics_matrix
