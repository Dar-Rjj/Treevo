import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-Scale Volatility-Momentum Integration
    # Price-Volatility Momentum Patterns
    df['vol_enhanced_momentum'] = ((df['close'] / df['close'].shift(3) - 1) - 
                                  (df['close'].shift(3) / df['close'].shift(6) - 1)) * df['true_range']
    
    df['vol_momentum_coherence'] = ((df['close'] / df['close'].shift(3) - 1) * 
                                   (df['close'].shift(3) / df['close'].shift(6) - 1) * df['true_range'])
    
    df['multi_timeframe_vol_momentum'] = ((df['close'] / df['close'].shift(7) - 1) - 
                                         (df['close'].shift(7) / df['close'].shift(21) - 1)) * df['true_range']
    
    # Volume-Volatility Entanglement
    df['volume_accel_vol'] = ((df['volume'] / df['volume'].shift(3) - 1) - 
                             (df['volume'].shift(3) / df['volume'].shift(6) - 1)) * df['true_range']
    
    # Volatility Concentration (5-day rolling)
    df['daily_range'] = df['high'] - df['low']
    df['vol_concentration'] = df['daily_range'] / df['daily_range'].rolling(window=5).mean()
    
    df['volume_vol_alignment'] = df['volume_accel_vol'] * df['vol_concentration']
    
    # Order Flow-Volatility Synthesis
    df['vwap'] = df['amount'] / df['volume']
    df['vol_order_efficiency'] = (df['open'] - df['close'].shift(1)) * df['vwap'] * df['true_range']
    
    df['vol_gap_efficiency'] = ((df['close'] - df['open']) / 
                               ((df['high'] - df['open']) + (df['open'] - df['low']))) * df['true_range']
    
    df['microstructure_vol_intensity'] = df['daily_range'] * df['volume'] / df['amount'] * df['true_range']
    
    # Asymmetric Volatility-Momentum Regimes
    df['up_day_vol_momentum'] = np.where(df['close'] > df['close'].shift(1),
                                        ((df['high'] - df['open']) / df['close'].shift(1)) * 
                                        (df['close'] / df['close'].shift(3) - 1), 0)
    
    df['down_day_vol_momentum'] = np.where(df['close'] < df['close'].shift(1),
                                          ((df['open'] - df['low']) / df['close'].shift(1)) * 
                                          (df['close'] / df['close'].shift(3) - 1), 0)
    
    df['vol_asymmetry_ratio'] = df['up_day_vol_momentum'] / (df['down_day_vol_momentum'] + 1e-8)
    
    # Asymmetry Persistence
    df['asymmetry_persistence'] = 0
    for i in range(1, len(df)):
        if df['vol_asymmetry_ratio'].iloc[i] > 1:
            df['asymmetry_persistence'].iloc[i] = df['asymmetry_persistence'].iloc[i-1] + 1
    
    # Volatility-Momentum Memory Patterns
    df['vol_momentum_accel'] = df['vol_enhanced_momentum'] - df['vol_enhanced_momentum'].shift(5)
    df['volume_vol_momentum_memory'] = df['volume_accel_vol'] - df['volume_accel_vol'].shift(5)
    
    # Volatility-Momentum Persistence
    df['vol_momentum_persistence'] = 0
    for i in range(1, len(df)):
        if df['vol_enhanced_momentum'].iloc[i] * df['vol_enhanced_momentum'].iloc[i-1] > 0:
            df['vol_momentum_persistence'].iloc[i] = df['vol_momentum_persistence'].iloc[i-1] + abs(df['vol_enhanced_momentum'].iloc[i])
    
    # Regime-Volatility Integration
    df['vol_momentum_interaction'] = df['vol_asymmetry_ratio'] * df['vol_momentum_coherence']
    df['asymmetry_momentum_alignment'] = df['asymmetry_persistence'] * df['vol_momentum_persistence']
    df['efficiency_regime_convergence'] = df['vol_gap_efficiency'] * df['vol_order_efficiency']
    
    # Microstructure Volatility Anchoring
    # Volume-Weighted Volatility Levels
    df['volume_weighted_high'] = (df['high'] * df['volume'] + 
                                 df['high'].shift(1) * df['volume'].shift(1) +
                                 df['high'].shift(2) * df['volume'].shift(2) +
                                 df['high'].shift(3) * df['volume'].shift(3) +
                                 df['high'].shift(4) * df['volume'].shift(4)) / \
                                (df['volume'] + df['volume'].shift(1) + df['volume'].shift(2) + 
                                 df['volume'].shift(3) + df['volume'].shift(4))
    
    df['volume_weighted_low'] = (df['low'] * df['volume'] + 
                                df['low'].shift(1) * df['volume'].shift(1) +
                                df['low'].shift(2) * df['volume'].shift(2) +
                                df['low'].shift(3) * df['volume'].shift(3) +
                                df['low'].shift(4) * df['volume'].shift(4)) / \
                               (df['volume'] + df['volume'].shift(1) + df['volume'].shift(2) + 
                                df['volume'].shift(3) + df['volume'].shift(4))
    
    df['volume_weighted_vol_range'] = df['volume_weighted_high'] - df['volume_weighted_low']
    df['vol_anchor_efficiency'] = df['volume_weighted_vol_range'] / (df['high'] - df['low'] + 1e-8)
    
    # Volatility Breakout Patterns
    df['high_vol_breakout'] = np.where(df['close'] > df['volume_weighted_high'],
                                      (df['close'] - df['volume_weighted_high']) * df['volume'] * df['true_range'], 0)
    
    df['low_vol_breakout'] = np.where(df['close'] < df['volume_weighted_low'],
                                     -(df['volume_weighted_low'] - df['close']) * df['volume'] * df['true_range'], 0)
    
    # Volatility Anchor Persistence
    df['vol_anchor_persistence'] = 0
    for i in range(1, len(df)):
        if (df['close'].iloc[i] > df['volume_weighted_high'].iloc[i]) == (df['close'].iloc[i-1] > df['volume_weighted_high'].iloc[i-1]):
            df['vol_anchor_persistence'].iloc[i] = df['vol_anchor_persistence'].iloc[i-1] + 1
    
    df['order_vol_anchor_alignment'] = df['vol_order_efficiency'] * df['vol_anchor_efficiency']
    
    # Microstructure-Volatility Integration
    df['volume_vol_confirmation'] = df['volume_accel_vol'] * (df['high_vol_breakout'] + df['low_vol_breakout'])
    df['anchor_vol_interaction'] = df['vol_anchor_efficiency'] * df['vol_concentration']
    df['behavioral_vol_momentum'] = df['vol_anchor_persistence'] * df['vol_momentum_persistence']
    
    # Adaptive Volatility-Regime Enhancement
    df['vol_asymmetry_weight'] = 1 + abs(df['vol_asymmetry_ratio'])
    df['vol_momentum_weight'] = 1 + abs(df['vol_momentum_coherence'])
    df['vol_anchor_weight'] = 1 + abs(df['vol_anchor_efficiency'])
    df['order_vol_weight'] = 1 + abs(df['vol_order_efficiency'])
    
    # Volatility Asymmetry Amplification
    df['vol_order_boost'] = df['vol_order_efficiency'] * (1 + abs(df['vol_asymmetry_ratio']))
    df['momentum_vol_enhancement'] = df['volume_vol_alignment'] * (1 + abs(df['vol_momentum_coherence']))
    df['anchor_microstructure_magnification'] = df['order_vol_anchor_alignment'] * (1 + df['vol_anchor_persistence'])
    
    # Dynamic Volatility Signal Selection
    df['vol_regime_choice'] = np.where(df['vol_asymmetry_ratio'] > 1, 
                                      df['vol_order_boost'], df['momentum_vol_enhancement'])
    
    df['momentum_triggered_switching'] = np.where(df['vol_momentum_coherence'] > 0,
                                                 df['anchor_microstructure_magnification'], 
                                                 df['efficiency_regime_convergence'])
    
    df['anchor_guided_selection'] = np.where(df['vol_anchor_persistence'] > 2,
                                           df['behavioral_vol_momentum'], 
                                           df['volume_vol_confirmation'])
    
    # Multi-Dimensional Volatility-Momentum Synthesis
    # Final factor combining all components
    df['vol_momentum_divergence'] = (df['vol_enhanced_momentum'] - df['multi_timeframe_vol_momentum']) * \
                                   df['vol_momentum_accel']
    
    df['volume_vol_momentum_distribution'] = df['volume_vol_alignment'] * df['microstructure_vol_intensity'] * \
                                           df['vol_concentration']
    
    df['memory_enhanced_signals'] = df['vol_momentum_persistence'] * df['volume_vol_momentum_memory'] * \
                                   df['asymmetry_persistence']
    
    # Final hierarchical volatility-momentum factor
    df['hierarchical_vol_momentum'] = (
        df['vol_regime_choice'] * 
        df['momentum_triggered_switching'] * 
        df['anchor_guided_selection'] *
        df['vol_momentum_divergence'] *
        df['volume_vol_momentum_distribution'] *
        df['memory_enhanced_signals'] *
        (1 + df['vol_asymmetry_weight']) *
        (1 + df['vol_momentum_weight']) *
        (1 + df['vol_anchor_weight'])
    )
    
    # Apply regime-dependent scaling
    high_vol_regime = df['true_range'] > df['true_range'].rolling(window=20).mean()
    low_vol_regime = df['true_range'] < df['true_range'].rolling(window=20).mean()
    
    df['final_factor'] = df['hierarchical_vol_momentum'].copy()
    df.loc[high_vol_regime, 'final_factor'] *= (1 + abs(df['volume_accel_vol']))
    df.loc[low_vol_regime, 'final_factor'] *= (1 + abs(df['vol_gap_efficiency']))
    
    return df['final_factor']
