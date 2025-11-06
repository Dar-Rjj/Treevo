import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Directional price persistence (trend strength)
    price_trend = (close - close.rolling(5).mean()) / close.rolling(5).std()
    trend_persistence = price_trend.rolling(8).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    
    # Volume-weighted mean reversion
    typical_price = (high + low + close) / 3
    price_deviation = (typical_price - typical_price.rolling(10).mean()) / typical_price.rolling(10).std()
    volume_weight = volume.rolling(5).mean() / volume.rolling(20).mean()
    volume_reversal = -price_deviation * volume_weight
    
    # Market regime classification
    volatility_regime = close.pct_change().rolling(10).std()
    high_vol_threshold = volatility_regime.rolling(30).quantile(0.7)
    low_vol_threshold = volatility_regime.rolling(30).quantile(0.3)
    
    # Regime-adaptive smoothing
    fast_momentum = close.pct_change(3)
    slow_momentum = close.pct_change(8)
    
    regime_signal = np.zeros_like(close)
    regime_condition_high = volatility_regime > high_vol_threshold
    regime_condition_low = volatility_regime < low_vol_threshold
    
    regime_signal[regime_condition_high] = fast_momentum[regime_condition_high]
    regime_signal[regime_condition_low] = slow_momentum[regime_condition_low]
    regime_signal[~(regime_condition_high | regime_condition_low)] = (fast_momentum + slow_momentum)[~(regime_condition_high | regime_condition_low)] / 2
    
    # Composite factor
    heuristics_matrix = 0.5 * trend_persistence + 0.3 * volume_reversal + 0.2 * regime_signal
    
    return heuristics_matrix
