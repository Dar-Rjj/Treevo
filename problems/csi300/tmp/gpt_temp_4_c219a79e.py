import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum with Volume Confirmation alpha factor
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Framework
    df['ultra_short_mom'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['short_term_mom'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['medium_term_mom'] = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    
    # Momentum Hierarchy Analysis
    df['mom_accel_short'] = df['ultra_short_mom'] / (df['short_term_mom'] + 1e-8)
    df['mom_accel_medium'] = df['short_term_mom'] / (df['medium_term_mom'] + 1e-8)
    
    # Volume Confirmation System
    df['volume_ratio_1d'] = df['volume'] / (df['volume'].shift(1) + 1e-8)
    df['volume_ratio_3d'] = df['volume'] / (df['volume'].shift(3) + 1e-8)
    
    # Volume-Momentum Alignment
    df['vol_mom_alignment_ultra'] = np.sign(df['volume_ratio_1d'] - 1) * np.sign(df['ultra_short_mom'])
    df['vol_mom_alignment_short'] = np.sign(df['volume_ratio_3d'] - 1) * np.sign(df['short_term_mom'])
    df['vol_mom_alignment_medium'] = np.sign(df['volume_ratio_3d'] - 1) * np.sign(df['medium_term_mom'])
    
    # Volume Regime Classification
    df['volume_regime'] = 'normal'
    df.loc[df['volume_ratio_3d'] > 1.5, 'volume_regime'] = 'high'
    df.loc[df['volume_ratio_3d'] < 0.7, 'volume_regime'] = 'low'
    
    # Market Regime Detection
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    # Volatility Regime
    df['volatility_regime'] = 'normal'
    df.loc[df['daily_range'] > 0.04, 'volatility_regime'] = 'high'
    df.loc[df['daily_range'] < 0.01, 'volatility_regime'] = 'low'
    
    # Trend Regime
    df['trend_regime'] = 'transition'
    df.loc[df['medium_term_mom'] > 0.05, 'trend_regime'] = 'strong_uptrend'
    df.loc[df['medium_term_mom'] < -0.05, 'trend_regime'] = 'strong_downtrend'
    df.loc[abs(df['medium_term_mom']) < 0.02, 'trend_regime'] = 'weak_trend'
    
    # Regime Persistence
    for regime in ['volatility_regime', 'trend_regime']:
        df[f'{regime}_persistence'] = 1
        for i in range(1, len(df)):
            if df[regime].iloc[i] == df[regime].iloc[i-1]:
                df[f'{regime}_persistence'].iloc[i] = df[f'{regime}_persistence'].iloc[i-1] + 1
    
    # Adaptive Signal Construction
    # Regime-Based Component Weighting
    df['weight_ultra'] = 0.3
    df['weight_short'] = 0.4
    df['weight_medium'] = 0.3
    
    # Adjust weights based on volatility regime
    df.loc[df['volatility_regime'] == 'high', 'weight_medium'] = 0.5
    df.loc[df['volatility_regime'] == 'high', 'weight_short'] = 0.3
    df.loc[df['volatility_regime'] == 'high', 'weight_ultra'] = 0.2
    
    df.loc[df['volatility_regime'] == 'low', 'weight_short'] = 0.5
    df.loc[df['volatility_regime'] == 'low', 'weight_ultra'] = 0.3
    df.loc[df['volatility_regime'] == 'low', 'weight_medium'] = 0.2
    
    # Adjust weights based on trend regime
    df.loc[df['trend_regime'] == 'strong_uptrend', 'weight_medium'] += 0.1
    df.loc[df['trend_regime'] == 'strong_downtrend', 'weight_medium'] += 0.1
    df.loc[df['trend_regime'] == 'weak_trend', 'weight_ultra'] += 0.1
    df.loc[df['trend_regime'] == 'weak_trend', 'weight_short'] += 0.1
    
    # Bounded Signal Components
    df['bounded_mom_ultra'] = np.tanh(df['ultra_short_mom'] * 10)
    df['bounded_mom_short'] = np.tanh(df['short_term_mom'] * 5)
    df['bounded_mom_medium'] = np.tanh(df['medium_term_mom'] * 3)
    
    df['bounded_accel_short'] = np.tanh(df['mom_accel_short'])
    df['bounded_accel_medium'] = np.tanh(df['mom_accel_medium'])
    
    # Volume alignment scores
    vol_alignment_weights = [0.3, 0.4, 0.3]  # ultra, short, medium
    df['volume_alignment_score'] = (
        df['vol_mom_alignment_ultra'] * vol_alignment_weights[0] +
        df['vol_mom_alignment_short'] * vol_alignment_weights[1] +
        df['vol_mom_alignment_medium'] * vol_alignment_weights[2]
    )
    
    # Volume-Confirmed Momentum
    df['confirmed_mom_ultra'] = df['bounded_mom_ultra'] * (1 + df['volume_alignment_score'] * 0.5)
    df['confirmed_mom_short'] = df['bounded_mom_short'] * (1 + df['volume_alignment_score'] * 0.5)
    df['confirmed_mom_medium'] = df['bounded_mom_medium'] * (1 + df['volume_alignment_score'] * 0.5)
    
    # Risk-Adjusted Integration
    # Volatility Scaling
    volatility_scale = 1 / (df['daily_range'] + 0.01)
    volatility_scale = np.clip(volatility_scale, 0.5, 2.0)  # Bound scaling
    
    # Regime Transition Smoothing
    regime_smooth = np.minimum(df['volatility_regime_persistence'], df['trend_regime_persistence']) / 10
    regime_smooth = np.clip(regime_smooth, 0.1, 1.0)
    
    # Multi-Timeframe Synthesis
    df['momentum_component'] = (
        df['confirmed_mom_ultra'] * df['weight_ultra'] +
        df['confirmed_mom_short'] * df['weight_short'] +
        df['confirmed_mom_medium'] * df['weight_medium']
    )
    
    df['acceleration_component'] = (
        df['bounded_accel_short'] * 0.6 +
        df['bounded_accel_medium'] * 0.4
    )
    
    # Alpha Factor Output
    df['alpha_factor'] = (
        df['momentum_component'] * 0.7 +
        df['acceleration_component'] * 0.3
    ) * volatility_scale * regime_smooth
    
    return df['alpha_factor']
