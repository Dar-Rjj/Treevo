import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Switching Volatility-Scaled Reversal with Volume Confirmation
    """
    data = df.copy()
    
    # Volatility-Scaled Price Reversal Component
    # Short-term reversal (3-day)
    short_reversal = (data['close'].shift(1) - data['close'].shift(3)) / data['close'].shift(3)
    short_vol = data['close'].pct_change().shift(1).rolling(window=5).std()
    short_scaled = short_reversal / (short_vol + 0.001)
    
    # Medium-term reversal (10-day)
    medium_reversal = (data['close'].shift(1) - data['close'].shift(10)) / data['close'].shift(10)
    medium_vol = data['close'].pct_change().shift(1).rolling(window=15).std()
    medium_scaled = medium_reversal / (medium_vol + 0.001)
    
    # Long-term reversal (20-day)
    long_reversal = (data['close'].shift(1) - data['close'].shift(20)) / data['close'].shift(20)
    long_vol = data['close'].pct_change().shift(1).rolling(window=25).std()
    long_scaled = long_reversal / (long_vol + 0.001)
    
    # Volume Regime Detection
    # Volume momentum
    vol_momentum_5d = data['volume'] / data['volume'].shift(5) - 1
    vol_momentum_10d = data['volume'] / data['volume'].shift(10) - 1
    
    # Volume regime classification
    high_vol_regime = vol_momentum_5d > vol_momentum_10d
    low_vol_regime = ~high_vol_regime
    
    # Volume volatility assessment
    vol_volatility = (data['volume'] / data['volume'].shift(1) - 1).rolling(window=10).std()
    vol_volatility_median = vol_volatility.rolling(window=60).median()
    high_vol_volatility = vol_volatility > vol_volatility_median
    low_vol_volatility = ~high_vol_volatility
    
    # Dynamic Threshold System
    # Price-based thresholds
    price_range = ((data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1)).rolling(window=20).mean()
    price_threshold = 0.5 * price_range
    
    # Volume-based thresholds
    volume_sma_20 = data['volume'].rolling(window=20).mean()
    volume_range = ((data['volume'].shift(1) - volume_sma_20.shift(2)) / volume_sma_20.shift(2)).abs().rolling(window=20).mean()
    volume_threshold = 0.3 * volume_range
    
    # Nonlinear Volume-Weighting
    raw_volume_weight = data['volume'] / volume_sma_20.shift(1)
    nonlinear_volume_weight = 2 / (1 + np.exp(-raw_volume_weight)) - 1
    
    # Regime-Adaptive Alpha Construction
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 25:  # Ensure enough data for calculations
            continue
            
        # Regime-based signal selection
        if high_vol_regime.iloc[i] and low_vol_volatility.iloc[i]:
            selected_reversal = short_scaled.iloc[i]
        elif low_vol_regime.iloc[i] and high_vol_volatility.iloc[i]:
            selected_reversal = long_scaled.iloc[i]
        else:
            selected_reversal = medium_scaled.iloc[i]
        
        # Dynamic threshold filtering
        price_thresh_val = price_threshold.iloc[i]
        vol_thresh_val = volume_threshold.iloc[i]
        vol_momentum_val = abs(vol_momentum_5d.iloc[i])
        
        # Apply thresholds
        if abs(selected_reversal) > price_thresh_val and vol_momentum_val > vol_thresh_val:
            volume_weight = nonlinear_volume_weight.iloc[i]
            alpha_factor.iloc[i] = selected_reversal * volume_weight
        else:
            alpha_factor.iloc[i] = 0
    
    return alpha_factor
