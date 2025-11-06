import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Dynamic Volatility-Normalized Momentum with Volume Divergence factor
    """
    df = data.copy()
    
    # Volatility-Normalized Momentum
    df['momentum'] = df['close'] / df['close'].shift(5) - 1
    df['volatility_proxy'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['normalized_momentum'] = df['momentum'] / df['volatility_proxy']
    
    # Volume Confirmation
    # Volume trend using linear regression slope
    def calc_volume_slope(volume_series):
        if len(volume_series) < 5 or volume_series.isna().any():
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = linregress(x, volume_series)
        return slope
    
    df['volume_trend'] = df['volume'].rolling(window=5).apply(
        calc_volume_slope, raw=False
    )
    
    # Volume strength
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_strength'] = df['volume'] / df['volume_ma_5']
    
    # Price-volume alignment
    df['volume_confirmation'] = (
        (df['momentum'] > 0) & (df['volume_trend'] > 0) |
        (df['momentum'] < 0) & (df['volume_trend'] < 0)
    )
    
    # Regime Detection
    # ATR calculation
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_20'] = df['tr'].rolling(window=20).mean()
    df['close_ma_20'] = df['close'].rolling(window=20).mean()
    df['volatility_regime'] = df['atr_20'] / df['close_ma_20']
    
    # Calculate universe median for regime classification
    volatility_median = df['volatility_regime'].median()
    df['high_vol_regime'] = df['volatility_regime'] > volatility_median
    
    # Adaptive Signal Combination
    # Volume multiplier
    df['volume_multiplier'] = np.where(
        df['volume_confirmation'],
        1 + df['volume_strength'],
        1 - df['volume_strength']
    )
    
    # Base signal with volume adjustment
    df['base_signal'] = df['normalized_momentum'] * df['volume_multiplier']
    
    # Regime adjustment
    df['final_signal'] = np.where(
        df['high_vol_regime'],
        df['base_signal'] * 0.7,  # Reduce momentum emphasis in high volatility
        df['base_signal'] * 1.3   # Enhance momentum emphasis in low volatility
    )
    
    return df['final_signal']
