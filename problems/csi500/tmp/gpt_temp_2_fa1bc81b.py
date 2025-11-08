import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Acceleration
    df['mom_accel_short'] = (df['close'] / df['close'].shift(3) - 1) - (df['close'] / df['close'].shift(8) - 1)
    df['mom_accel_medium'] = (df['close'] / df['close'].shift(8) - 1) - (df['close'] / df['close'].shift(20) - 1)
    df['mom_persistence'] = df['mom_accel_short'] - df['mom_accel_short'].shift(1)
    
    # Volume Acceleration
    df['vol_momentum'] = (df['volume'] / df['volume'].shift(3)) - (df['volume'] / df['volume'].shift(8))
    df['vol_price_alignment'] = np.sign(df['mom_accel_short']) * np.sign(df['vol_momentum'])
    
    # Microstructure Efficiency
    df['price_efficiency'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    df['opening_gap_efficiency'] = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Behavioral Anchoring
    df['high_20'] = df['high'].rolling(window=20, min_periods=10).max()
    df['low_20'] = df['low'].rolling(window=20, min_periods=10).min()
    df['proximity_extremes'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
    df['opening_rejection'] = np.abs(df['open'] - df['high']) / (df['high'] - df['low'])
    
    # Volatility Regime
    df['high_10'] = df['high'].rolling(window=10, min_periods=5).max()
    df['low_10'] = df['low'].rolling(window=10, min_periods=5).min()
    df['close_10'] = df['close'].rolling(window=10, min_periods=5).mean()
    df['range_volatility'] = (df['high_10'] - df['low_10']) / df['close_10']
    
    df['true_range'] = np.maximum(
        np.maximum(df['high'] - df['low'], 
                  np.abs(df['high'] - df['close'].shift(1))),
        np.abs(df['low'] - df['close'].shift(1))
    )
    df['true_range_volatility'] = df['true_range'].rolling(window=10, min_periods=5).mean()
    
    # Volatility regime classification
    vol_quantiles = df['range_volatility'].rolling(window=50, min_periods=25).apply(
        lambda x: pd.qcut(x, q=[0, 0.3, 0.7, 1.0], labels=False, duplicates='drop').iloc[-1] 
        if len(x.dropna()) >= 25 else 1, raw=False
    )
    
    # Acceleration-Efficiency Divergence
    df['acceleration_strength'] = (df['mom_accel_short'] + df['mom_accel_medium']) / 2
    df['efficiency_score'] = (df['price_efficiency'].abs() + (1 - df['opening_rejection'])) / 2
    
    # Divergence scoring
    df['strong_accel_poor_eff'] = df['acceleration_strength'] * (1 - df['efficiency_score'])
    df['weak_accel_strong_eff'] = (1 - df['acceleration_strength'].abs()) * df['efficiency_score']
    
    # Efficiency-Weighted Acceleration
    df['eff_weighted_accel'] = df['acceleration_strength'] * df['efficiency_score'] * df['vol_price_alignment']
    
    # Regime-Adaptive Combination
    high_vol_weight = np.where(vol_quantiles == 2, 0.3, 0.6)  # Reduce acceleration in high vol
    low_vol_weight = np.where(vol_quantiles == 0, 0.8, 0.6)   # Amplify acceleration in low vol
    
    df['regime_weighted_accel'] = df['eff_weighted_accel'] * np.where(
        vol_quantiles == 2, high_vol_weight, 
        np.where(vol_quantiles == 0, low_vol_weight, 0.6)
    )
    
    # Regime-specific divergence emphasis
    df['divergence_component'] = np.where(
        vol_quantiles == 2, df['weak_accel_strong_eff'],  # High vol: prefer efficiency
        np.where(vol_quantiles == 0, df['strong_accel_poor_eff'],  # Low vol: prefer acceleration
                (df['strong_accel_poor_eff'] + df['weak_accel_strong_eff']) / 2)  # Medium: balanced
    )
    
    # Final factor combination
    df['raw_factor'] = (
        0.4 * df['regime_weighted_accel'] +
        0.3 * df['divergence_component'] +
        0.2 * df['mom_persistence'] +
        0.1 * df['proximity_extremes']
    )
    
    # Volatility regime conditioning
    final_factor = df['raw_factor'] * np.where(
        vol_quantiles == 2, 0.7,  # Scale down in high volatility
        np.where(vol_quantiles == 0, 1.2, 1.0)  # Amplify in low volatility
    )
    
    return final_factor
