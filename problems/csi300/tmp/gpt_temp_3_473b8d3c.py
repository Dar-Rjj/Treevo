import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Simple Momentum
    n = 10  # Lookback period for momentum
    simple_momentum = df['close'].pct_change(n)
    
    # Volume Adjusted Component
    volume_change = df['volume'].pct_change()
    volume_adjusted_momentum = simple_momentum * volume_change
    
    # Price Reversal Sensitivity
    high_low_spread = df['high'] - df['low']
    price_reversal_sensitivity = high_low_spread * (df['volume'] / df['volume'].mean())
    
    # Combine Components
    intermediate_alpha_factor = volume_adjusted_momentum - price_reversal_sensitivity
    
    # Incorporate Enhanced Price Gaps
    gap_oc = df['open'] - df['close'].shift(1)
    gap_hl = df['high'] - df['low']
    combined_momentum = intermediate_alpha_factor + gap_oc + gap_hl
    
    # Confirm with Volume
    vol_ma5 = df['volume'].rolling(window=5).mean()
    vol_ma20 = df['volume'].rolling(window=20).mean()
    confirmed_momentum = combined_momentum * np.where(vol_ma5 > vol_ma20, 1.2, 0.8)
    
    # Adjust by ATR
    true_range = np.maximum.reduce([df['high'] - df['low'], 
                                    abs(df['high'] - df['close'].shift(1)), 
                                    abs(df['low'] - df['close'].shift(1))])
    atr = true_range.rolling(window=14).mean()
    adj_momentum_atr = confirmed_momentum / atr
    
    # Final Adjustment
    vol_ma_20 = df['volume'].rolling(window=20).mean()
    final_factor = np.where(vol_ma_20 > 1.5 * vol_ma_20.mean(), 
                            adj_momentum_atr * vol_ma_20 * 0.9, 
                            adj_momentum_atr * vol_ma_20)
    
    return pd.Series(final_factor, index=df.index, name='FinalFactor')
