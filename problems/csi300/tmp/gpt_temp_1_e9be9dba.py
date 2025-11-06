import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    df = data.copy()
    
    # Volatility-Normalized Momentum
    # 5-day price momentum
    momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # 5-day average true range
    atr = (df['high'] - df['low']).rolling(window=5).mean()
    
    # Normalize momentum by volatility
    vol_normalized_momentum = momentum / atr
    
    # Volume-Price Divergence
    volume_slope = pd.Series(index=df.index, dtype=float)
    price_slope = pd.Series(index=df.index, dtype=float)
    divergence = pd.Series(index=df.index, dtype=bool)
    
    for i in range(4, len(df)):
        if i >= 4:
            # Volume slope (5-day window)
            vol_window = df['volume'].iloc[i-4:i+1]
            vol_slope_val = linregress(range(5), vol_window.values)[0]
            volume_slope.iloc[i] = vol_slope_val
            
            # Price slope (5-day window)
            price_window = df['close'].iloc[i-4:i+1]
            price_slope_val = linregress(range(5), price_window.values)[0]
            price_slope.iloc[i] = price_slope_val
            
            # Detect divergence
            divergence.iloc[i] = np.sign(vol_slope_val) != np.sign(price_slope_val)
    
    # Regime Detection
    # 20-day volatility
    volatility = df['close'].rolling(window=20).std()
    
    # 252-day median volatility
    median_volatility = volatility.rolling(window=252).median()
    
    # Volatility regime
    high_vol_regime = volatility > median_volatility
    low_vol_regime = volatility <= median_volatility
    
    # Alpha Combination
    base_signal = vol_normalized_momentum
    
    # Volume-confirmed signal
    volume_confirmed_signal = base_signal * (1 + abs(volume_slope))
    
    # Regime weighting
    regime_weight = pd.Series(index=df.index, dtype=float)
    regime_weight[high_vol_regime] = -1.0  # Mean reversion in high volatility
    regime_weight[low_vol_regime] = 1.0    # Momentum continuation in low volatility
    
    # Final alpha
    final_alpha = pd.Series(index=df.index, dtype=float)
    final_alpha[divergence] = regime_weight[divergence] * volume_confirmed_signal[divergence]
    final_alpha[~divergence] = regime_weight[~divergence] * base_signal[~divergence]
    
    return final_alpha
