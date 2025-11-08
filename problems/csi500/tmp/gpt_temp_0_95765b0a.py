import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum Factor
    Combines multi-timeframe momentum with volume confirmation and volatility regime detection
    """
    
    # Multi-Timeframe Momentum Analysis
    # Short-Term Momentum (5-day)
    df['mom_5d'] = df['close'] / df['close'].shift(5) - 1
    df['mom_persistence_5d'] = (df['mom_5d'].shift(1) * df['mom_5d'].shift(2) * df['mom_5d'].shift(3) > 0).astype(int)
    
    # Medium-Term Momentum (20-day)
    df['mom_20d'] = df['close'] / df['close'].shift(20) - 1
    df['mom_accel_20d'] = df['mom_20d'] - df['mom_20d'].shift(5)
    
    # Long-Term Momentum (60-day)
    df['mom_60d'] = df['close'] / df['close'].shift(60) - 1
    df['trend_direction'] = np.sign(df['mom_60d'])
    
    # Volume Confirmation Framework
    # Volume Trend Analysis
    df['volume_5d_avg'] = df['volume'].rolling(5).mean()
    df['volume_20d_avg'] = df['volume'].rolling(20).mean()
    df['volume_ratio_5_20'] = df['volume_5d_avg'] / df['volume_20d_avg']
    
    volume_median = df['volume'].rolling(60).median()
    df['volume_breakout'] = (df['volume'] > volume_median * 1.2).astype(int)
    
    # Volume-Price Alignment
    df['vwap_5d'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
    df['volume_weighted_return'] = (df['vwap_5d'] / df['vwap_5d'].shift(5) - 1) * df['volume_ratio_5_20']
    
    df['volume_persistence'] = ((df['mom_5d'] > 0) & (df['volume_ratio_5_20'] > 1) | 
                               (df['mom_5d'] < 0) & (df['volume_ratio_5_20'] < 0.8)).astype(int)
    
    # Volume Regime Classification
    df['high_volume_trending'] = ((df['volume_ratio_5_20'] > 1.2) & 
                                 (df['mom_5d'].abs() > df['mom_5d'].rolling(20).std())).astype(int)
    df['low_volume_consolidation'] = ((df['volume_ratio_5_20'] < 0.8) & 
                                     (df['mom_5d'].abs() < df['mom_5d'].rolling(20).std() * 0.5)).astype(int)
    
    # Volatility Regime Detection
    # Multi-Scale Volatility Measurement
    df['volatility_10d'] = df['close'].pct_change().rolling(10).std()
    df['range_volatility_20d'] = ((df['high'] - df['low']) / df['close']).rolling(20).mean()
    df['composite_volatility'] = (df['volatility_10d'] + df['range_volatility_20d']) / 2
    
    # Volatility Regime Classification
    vol_median = df['composite_volatility'].rolling(60).median()
    df['high_vol_regime'] = (df['composite_volatility'] > vol_median).astype(int)
    df['low_vol_regime'] = (df['composite_volatility'] <= vol_median).astype(int)
    
    # Volatility-Adjusted Signals
    df['mom_5d_vol_adj'] = df['mom_5d'] / (df['composite_volatility'] + 1e-8)
    df['mom_20d_vol_adj'] = df['mom_20d'] / (df['composite_volatility'] + 1e-8)
    
    # Regime persistence (maintain regime for minimum 5 days)
    df['regime_persistence'] = df['high_vol_regime'].rolling(5).sum()
    df['stable_high_vol'] = (df['regime_persistence'] == 5).astype(int)
    df['stable_low_vol'] = (df['regime_persistence'] == 0).astype(int)
    
    # Adaptive Factor Construction
    # High volatility regime signals
    high_vol_momentum = (df['mom_5d_vol_adj'] * 0.6 + 
                        df['volume_weighted_return'] * 0.4) * df['volume_persistence']
    
    # Low volatility regime signals  
    low_vol_momentum = (df['mom_20d_vol_adj'] * 0.7 + 
                       df['mom_accel_20d'] * 0.3) * df['trend_direction']
    
    # Regime-Weighted Combination
    regime_weighted_signal = (high_vol_momentum * df['stable_high_vol'] + 
                             low_vol_momentum * df['stable_low_vol'])
    
    # Cross-Timeframe Validation
    timeframe_alignment = ((np.sign(df['mom_5d']) == np.sign(df['mom_20d'])).astype(int) + 
                          (np.sign(df['mom_20d']) == np.sign(df['mom_60d'])).astype(int))
    
    valid_alignment = (timeframe_alignment >= 1).astype(int)
    
    # Volume confirmation
    volume_confirmation = ((df['mom_5d'] > 0) & (df['volume_ratio_5_20'] > 1) | 
                          (df['mom_5d'] < 0) & (df['volume_ratio_5_20'] < 1)).astype(int)
    
    # Final Alpha Signal Generation with momentum decay
    decay_factor = 0.95
    df['raw_factor'] = (regime_weighted_signal * valid_alignment * volume_confirmation)
    
    # Apply momentum decay for stale signals
    df['factor_decayed'] = df['raw_factor']
    for i in range(1, len(df)):
        if df['raw_factor'].iloc[i] == 0:
            df['factor_decayed'].iloc[i] = df['factor_decayed'].iloc[i-1] * decay_factor
    
    return df['factor_decayed']
