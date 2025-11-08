import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Price Acceleration: (Close_t - Close_t-3)/Close_t-3 - (Close_t - Close_t-8)/Close_t-8
    price_accel_3 = df['close'].pct_change(periods=3)
    price_accel_8 = df['close'].pct_change(periods=8)
    price_acceleration = price_accel_3 - price_accel_8
    
    # Volume Acceleration: (Volume_t - Volume_t-3)/Volume_t-3 - (Volume_t - Volume_t-8)/Volume_t-8
    volume_accel_3 = df['volume'].pct_change(periods=3)
    volume_accel_8 = df['volume'].pct_change(periods=8)
    volume_acceleration = volume_accel_3 - volume_accel_8
    
    # Price-Volume Divergence
    pv_divergence = price_acceleration - volume_acceleration
    
    # Volatility Asymmetry
    upside_volatility = returns.rolling(window=7, min_periods=5).apply(
        lambda x: x[x > 0].std() if len(x[x > 0]) >= 3 else 0
    )
    
    downside_volatility = returns.rolling(window=7, min_periods=5).apply(
        lambda x: x[x < 0].std() if len(x[x < 0]) >= 3 else 0
    )
    
    # Handle NaN values in volatility measures
    upside_volatility = upside_volatility.fillna(0)
    downside_volatility = downside_volatility.fillna(0)
    
    # Alpha Construction
    volatility_asymmetry = upside_volatility - downside_volatility
    sign_pv_divergence = np.sign(pv_divergence)
    
    # Final Factor: (Price Acceleration - Volume Acceleration) × (1 + (Upside Volatility - Downside Volatility) × sign(Price Acceleration - Volume Acceleration))
    factor = pv_divergence * (1 + volatility_asymmetry * sign_pv_divergence)
    
    return factor
