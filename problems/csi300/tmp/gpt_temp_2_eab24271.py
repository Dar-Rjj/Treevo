import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Momentum Acceleration with Volume-Volatility Confirmation factor
    """
    # Multi-Timeframe Momentum Analysis
    df = df.copy()
    
    # Calculate price returns for different timeframes
    df['ret_3d'] = df['close'] / df['close'].shift(3) - 1
    df['ret_8d'] = df['close'] / df['close'].shift(8) - 1
    df['ret_21d'] = df['close'] / df['close'].shift(21) - 1
    
    # Momentum acceleration differences
    df['mom_accel_3_8'] = df['ret_3d'] - df['ret_8d']
    df['mom_accel_8_21'] = df['ret_8d'] - df['ret_21d']
    
    # Momentum curvature (second-order differences)
    df['mom_curvature'] = df['mom_accel_3_8'] - df['mom_accel_8_21']
    
    # Volume and Amount Confirmation
    df['vol_mom_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['vol_mom_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['vol_accel'] = df['vol_mom_5d'] - df['vol_mom_10d']
    
    # Price-volume alignment
    df['price_vol_alignment_5d'] = df['ret_3d'] * df['vol_mom_5d']
    df['price_vol_alignment_10d'] = df['ret_8d'] * df['vol_mom_10d']
    
    # Amount-based metrics
    df['amount_mom_5d'] = df['amount'] / df['amount'].shift(5) - 1
    df['vol_to_amount_ratio'] = df['volume'] / df['amount']
    df['vol_amount_ratio_mom'] = df['vol_to_amount_ratio'] / df['vol_to_amount_ratio'].shift(5) - 1
    
    # Volatility Context Assessment
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['volatility_5d'] = df['daily_range'].rolling(window=5).mean()
    df['volatility_10d'] = df['daily_range'].rolling(window=10).mean()
    
    # Volatility momentum and acceleration
    df['vol_mom'] = df['volatility_5d'] / df['volatility_10d'] - 1
    df['vol_accel'] = df['volatility_5d'] - df['volatility_10d']
    
    # Volatility regime classification
    df['vol_regime'] = pd.cut(df['volatility_10d'], 
                             bins=[-np.inf, df['volatility_10d'].quantile(0.33), 
                                  df['volatility_10d'].quantile(0.66), np.inf],
                             labels=[0, 1, 2])
    
    # Volatility-adjusted momentum
    df['vol_adj_mom_3d'] = df['ret_3d'] / (df['volatility_5d'] + 1e-8)
    df['vol_adj_mom_8d'] = df['ret_8d'] / (df['volatility_10d'] + 1e-8)
    
    # Pattern Recognition
    # Momentum-volume divergence
    df['mom_vol_divergence_5d'] = np.sign(df['ret_3d']) != np.sign(df['vol_mom_5d'])
    df['mom_vol_divergence_10d'] = np.sign(df['ret_8d']) != np.sign(df['vol_mom_10d'])
    
    # Signal strength across timeframes
    df['signal_strength'] = (df['ret_3d'].abs() + df['ret_8d'].abs() + df['ret_21d'].abs()) / 3
    
    # Momentum regime transitions
    df['mom_regime_change'] = ((df['ret_3d'] > 0) != (df['ret_21d'] > 0)).astype(int)
    
    # Composite Factor Generation
    # Base momentum acceleration factor
    base_factor = (df['mom_curvature'] * 0.4 + 
                   df['vol_adj_mom_3d'] * 0.3 + 
                   df['vol_adj_mom_8d'] * 0.3)
    
    # Volume confirmation multiplier
    volume_confirmation = (1 + 
                          df['price_vol_alignment_5d'] * 0.2 + 
                          df['price_vol_alignment_10d'] * 0.15 + 
                          df['vol_amount_ratio_mom'] * 0.1)
    
    # Pattern recognition signals
    pattern_multiplier = 1.0
    pattern_multiplier += df['mom_regime_change'] * 0.1
    pattern_multiplier -= df['mom_vol_divergence_5d'].astype(float) * 0.05
    pattern_multiplier -= df['mom_vol_divergence_10d'].astype(float) * 0.05
    
    # Volatility regime weighting
    regime_coefficients = {
        0: 1.2,  # Low volatility - amplify signals
        1: 1.0,  # Medium volatility - neutral
        2: 0.8   # High volatility - dampen signals
    }
    
    # Apply regime-specific coefficients
    regime_weight = df['vol_regime'].map(regime_coefficients).fillna(1.0)
    
    # Final composite factor
    final_factor = (base_factor * volume_confirmation * pattern_multiplier * 
                   regime_weight * df['signal_strength'])
    
    # Clean and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = final_factor.fillna(0)
    
    return final_factor
