import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate regime-adaptive momentum-volume convergence alpha factor
    """
    # Multi-Timeframe Momentum Dynamics
    df['momentum_2d'] = df['close'] / df['close'].shift(2) - 1
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    
    df['momentum_acceleration'] = df['momentum_2d'] - df['momentum_5d']
    df['trend_strength'] = df['momentum_5d'] - df['momentum_10d']
    df['momentum_consistency'] = np.sign(df['momentum_2d']) * np.sign(df['momentum_5d']) * np.sign(df['momentum_10d'])
    
    # Volume-Price Synchronization Framework
    df['volume_momentum_2d'] = df['volume'] / df['volume'].shift(2)
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5)
    df['volume_acceleration'] = df['volume_momentum_2d'] - df['volume_momentum_5d']
    
    # Price-volume alignment
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['direction_alignment'] = np.sign(df['price_change']) * np.sign(df['volume_change'])
    df['magnitude_alignment'] = np.minimum(np.abs(df['price_change']), np.abs(df['volume_change']))
    
    # Rolling consistency score
    df['aligned_day'] = (df['direction_alignment'] > 0).astype(int)
    df['consistency_score'] = df['aligned_day'].rolling(window=5, min_periods=3).mean()
    df['sync_strength'] = df['direction_alignment'] * df['magnitude_alignment'] * df['consistency_score']
    
    # Volatility Regime Adaptive System
    # True range calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = np.abs(df['high'] - df['close'].shift(1))
    df['tr3'] = np.abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling volatility
    df['returns'] = df['close'].pct_change()
    df['rolling_volatility'] = df['returns'].rolling(window=5, min_periods=3).std()
    
    # Regime classification using 20-day rolling percentiles
    df['vol_percentile'] = df['rolling_volatility'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    df['vol_regime'] = 'normal'
    df.loc[df['vol_percentile'] > 0.8, 'vol_regime'] = 'high'
    df.loc[df['vol_percentile'] < 0.2, 'vol_regime'] = 'low'
    
    # Regime-specific thresholds
    conditions = [
        df['vol_regime'] == 'high',
        df['vol_regime'] == 'normal', 
        df['vol_regime'] == 'low'
    ]
    choices_momentum = [0.03, 0.02, 0.01]
    choices_volume = [1.3, 1.2, 1.1]
    
    df['momentum_threshold'] = np.select(conditions, choices_momentum, default=0.02)
    df['volume_threshold'] = np.select(conditions, choices_volume, default=1.2)
    
    # Momentum-Volume Convergence Analysis
    df['convergence_detected'] = (
        (np.sign(df['momentum_acceleration']) == np.sign(df['volume_acceleration'])) &
        (np.abs(df['momentum_acceleration']) > df['momentum_threshold']) &
        (df['volume_acceleration'] > df['volume_threshold']) &
        (df['sync_strength'] > 0.5)
    )
    
    # Convergence strength scoring
    df['base_score'] = df['momentum_acceleration'] * df['volume_acceleration']
    df['alignment_multiplier'] = df['sync_strength']
    
    # Regime multiplier
    regime_multiplier_conditions = [
        df['vol_regime'] == 'high',
        df['vol_regime'] == 'normal',
        df['vol_regime'] == 'low'
    ]
    regime_multiplier_choices = [0.7, 1.0, 1.3]
    df['regime_multiplier'] = np.select(regime_multiplier_conditions, regime_multiplier_choices, default=1.0)
    
    # Regime-Adaptive Signal Construction
    df['raw_signal'] = df['base_score'] * df['alignment_multiplier'] * df['regime_multiplier']
    
    # Apply convergence detection filter
    df['filtered_signal'] = df['raw_signal'] * df['convergence_detected']
    
    # Risk adjustment - volatility scaling
    df['volatility_scaling'] = 1 / (1 + df['rolling_volatility'].rolling(window=10, min_periods=5).mean())
    df['risk_adjusted_signal'] = df['filtered_signal'] * df['volatility_scaling']
    
    # Final alpha factor with position constraints
    df['alpha_factor'] = np.clip(df['risk_adjusted_signal'], -2.0, 2.0)
    
    # Clean up intermediate columns
    result = df['alpha_factor'].copy()
    
    return result
