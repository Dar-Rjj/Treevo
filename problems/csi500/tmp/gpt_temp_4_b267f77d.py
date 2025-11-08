import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Switching Volatility-Scaled Reversal with Volume Confirmation
    """
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Calculate ATR for different periods
    def calculate_atr(period):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    atr_3 = calculate_atr(3)
    atr_8 = calculate_atr(8)
    atr_15 = calculate_atr(15)
    
    # Price Reversal Components
    reversal_3d = (df['close'].shift(3) - df['close']) / atr_3
    reversal_8d = (df['close'].shift(8) - df['close']) / atr_8
    reversal_15d = (df['close'].shift(15) - df['close']) / atr_15
    
    # Volatility Regime Detection
    vol_5d = returns.rolling(window=5).std()
    vol_20d = returns.rolling(window=20).std()
    vol_ratio = vol_5d / vol_20d
    
    # Regime Classification
    high_vol_regime = vol_ratio > 1.2
    low_vol_regime = vol_ratio < 0.8
    normal_vol_regime = (vol_ratio >= 0.8) & (vol_ratio <= 1.2)
    
    # Volume Confirmation System
    volume_3d_momentum = df['volume'] / df['volume'].shift(3) - 1
    volume_8d_momentum = df['volume'] / df['volume'].shift(8) - 1
    volume_divergence = (volume_3d_momentum + volume_8d_momentum) / 2
    
    # Dynamic threshold for volume divergence
    volume_threshold = volume_divergence.rolling(window=252).apply(
        lambda x: np.percentile(np.abs(x.dropna()), 20), raw=False
    ).shift(1)
    
    # Volume weight using tanh
    volume_weight = np.tanh(volume_divergence * 2)
    
    # Alpha Factor Construction
    # Regime-based reversal selection
    selected_reversal = pd.Series(index=df.index, dtype=float)
    selected_reversal[high_vol_regime] = reversal_3d[high_vol_regime]
    selected_reversal[low_vol_regime] = reversal_15d[low_vol_regime]
    selected_reversal[normal_vol_regime] = reversal_8d[normal_vol_regime]
    
    # Volatility scaling
    volatility_scaled_reversal = selected_reversal / vol_20d
    
    # Volume confirmation filter
    volume_confirmed_factor = volatility_scaled_reversal * volume_weight
    
    # Dynamic threshold enforcement
    factor_abs = np.abs(volume_confirmed_factor)
    threshold = factor_abs.rolling(window=252).apply(
        lambda x: np.percentile(x.dropna(), 30), raw=False
    ).shift(1)
    
    # Apply threshold
    final_factor = volume_confirmed_factor.copy()
    weak_signals = factor_abs <= threshold
    final_factor[weak_signals] = 0
    
    return final_factor
