import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Price trend persistence component
    trend_strength = (close - close.rolling(window=10).mean()) / close.rolling(window=10).std()
    trend_persistence = trend_strength.rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Liquidity regime detection
    volume_trend = volume.rolling(window=8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volatility_regime = (high - low).rolling(window=8).std()
    liquidity_volatility_interaction = volume_trend * volatility_regime
    
    # Regime-switching factor
    regime_threshold = liquidity_volatility_interaction.rolling(window=20).median()
    regime_signal = np.where(liquidity_volatility_interaction > regime_threshold, 1, -1)
    
    heuristics_matrix = trend_persistence * regime_signal
    
    return heuristics_matrix
