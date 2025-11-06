import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Normalized Price-Volume Convergence Divergence factor
    """
    df = data.copy()
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate 10-day average True Range for price volatility
    avg_true_range = true_range.rolling(window=10, min_periods=1).mean()
    
    # Calculate rolling standard deviation of volume for volume volatility
    volume_volatility = df['volume'].rolling(window=10, min_periods=1).std()
    
    # Calculate rolling correlation between price changes and volume changes (10-day window)
    price_change = df['close'].pct_change()
    volume_change = df['volume'].pct_change()
    price_volume_corr = price_change.rolling(window=10, min_periods=1).corr(volume_change)
    
    # Initialize result series
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        if i < 20:
            continue
            
        # Short-term price momentum (5-day)
        st_price_momentum = df['close'].iloc[i] - df['close'].iloc[i-5]
        # Medium-term price momentum (20-day)
        mt_price_momentum = df['close'].iloc[i] - df['close'].iloc[i-20]
        
        # Short-term volume momentum (5-day)
        st_volume_momentum = df['volume'].iloc[i] - df['volume'].iloc[i-5]
        # Medium-term volume momentum (20-day)
        mt_volume_momentum = df['volume'].iloc[i] - df['volume'].iloc[i-20]
        
        # Normalize price momentum by price volatility
        if avg_true_range.iloc[i] > 0:
            st_norm_price_mom = st_price_momentum / avg_true_range.iloc[i]
            mt_norm_price_mom = mt_price_momentum / avg_true_range.iloc[i]
        else:
            st_norm_price_mom = 0
            mt_norm_price_mom = 0
        
        # Normalize volume momentum by volume volatility
        if volume_volatility.iloc[i] > 0:
            st_norm_volume_mom = st_volume_momentum / volume_volatility.iloc[i]
            mt_norm_volume_mom = mt_volume_momentum / volume_volatility.iloc[i]
        else:
            st_norm_volume_mom = 0
            mt_norm_volume_mom = 0
        
        # Compute divergence signals
        st_divergence = st_norm_price_mom - st_norm_volume_mom
        mt_divergence = mt_norm_price_mom - mt_norm_volume_mom
        
        # Get correlation strength (absolute value)
        corr_strength = abs(price_volume_corr.iloc[i]) if not pd.isna(price_volume_corr.iloc[i]) else 0
        
        # Weight divergences by correlation strength
        st_weighted_div = st_divergence * corr_strength
        mt_weighted_div = mt_divergence * corr_strength
        
        # Combine timeframes with correlation persistence weighting
        # Use average of recent correlations as persistence measure
        recent_corr_persistence = price_volume_corr.iloc[max(i-4, 0):i+1].abs().mean() if i >= 4 else corr_strength
        
        if not pd.isna(recent_corr_persistence):
            # Higher weight to medium-term when correlation is persistent
            mt_weight = min(recent_corr_persistence, 1.0)
            st_weight = 1.0 - mt_weight
        else:
            st_weight = 0.5
            mt_weight = 0.5
        
        # Final alpha factor value
        alpha_factor.iloc[i] = (st_weighted_div * st_weight + mt_weighted_div * mt_weight)
    
    # Fill NaN values with 0
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
