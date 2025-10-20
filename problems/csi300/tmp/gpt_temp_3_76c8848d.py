import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Parameters
    N = 14  # True Range window
    M = 20  # Volatility rolling window
    P = 50  # Volatility regime comparison window
    Q = 21  # Volume median window
    R = 5   # Volume persistence window
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Volatility State Detection
    # True Range calculation
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Rolling volatility using True Range standard deviation
    rolling_volatility = true_range.rolling(window=M, min_periods=1).std()
    
    # Volatility regime classification
    vol_median = rolling_volatility.rolling(window=P, min_periods=1).median()
    high_vol_regime = rolling_volatility > vol_median
    
    # Price Reversal Strength
    # Price reversal magnitude
    reversal_magnitude = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Reversal persistence
    reversal_sign = np.sign(reversal_magnitude)
    reversal_persistence = reversal_sign.groupby(reversal_sign.ne(reversal_sign.shift()).cumsum()).cumcount() + 1
    reversal_persistence = reversal_persistence * reversal_sign
    
    # Reversal intensity
    reversal_intensity = abs(reversal_magnitude) * df['volume']
    
    # Volume Confirmation Analysis
    # Abnormal volume
    vol_median_rolling = df['volume'].rolling(window=Q, min_periods=1).median()
    abnormal_volume = df['volume'] / (vol_median_rolling + 1e-8)
    
    # Volume-price alignment
    price_change = np.sign(df['close'] - df['open'])
    volume_change = np.sign(df['volume'] - df['volume'].shift(1))
    volume_price_alignment = price_change * volume_change
    
    # Volume persistence
    volume_persistence = volume_price_alignment.rolling(window=R, min_periods=1).sum()
    
    # Adaptive Signal Generation
    for i in range(len(df)):
        if i < max(N, M, P, Q, R):
            result.iloc[i] = 0
            continue
            
        current_high_vol = high_vol_regime.iloc[i]
        current_reversal_mag = reversal_magnitude.iloc[i]
        current_reversal_pers = reversal_persistence.iloc[i]
        current_reversal_int = reversal_intensity.iloc[i]
        current_abnormal_vol = abnormal_volume.iloc[i]
        current_vol_pers = volume_persistence.iloc[i]
        
        if current_high_vol:
            # High volatility regime
            if abs(current_reversal_mag) > 0.3 and current_abnormal_vol > 1.5:
                # Strong reversal with high volume → contrarian signal
                signal = -np.sign(current_reversal_mag) * min(abs(current_reversal_mag), 1.0)
            else:
                # Weak reversal with low volume → neutral signal
                signal = 0
        else:
            # Low volatility regime
            if abs(current_reversal_pers) >= 3 and current_vol_pers > 0:
                # Persistent reversal with volume confirmation → momentum signal
                signal = np.sign(current_reversal_pers) * min(abs(current_reversal_mag), 0.5)
            else:
                # Inconsistent volume-price alignment → mean reversion signal
                signal = -np.sign(current_reversal_mag) * min(abs(current_reversal_mag), 0.3)
        
        # Scale by reversal intensity and volume persistence
        signal *= min(current_reversal_int / (df['volume'].iloc[:i+1].mean() + 1e-8), 2.0)
        signal *= (1 + min(current_vol_pers / R, 1.0))
        
        result.iloc[i] = signal
    
    return result
