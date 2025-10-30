import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Period Momentum Elasticity Analysis
    df['short_term_momentum_eff'] = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min())
    df['medium_term_momentum_eff'] = (df['close'] - df['close'].shift(10)) / (df['high'].rolling(10).max() - df['low'].rolling(10).min())
    df['long_term_momentum_eff'] = (df['close'] - df['close'].shift(20)) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
    
    df['momentum_elasticity_div'] = df['short_term_momentum_eff'] * df['medium_term_momentum_eff'] * df['long_term_momentum_eff']
    df['momentum_convergence'] = df['short_term_momentum_eff'] * df['medium_term_momentum_eff']
    
    # Volume-Pressure Regime Classification
    df['volume_shock_intensity'] = df['volume'] / df['volume'].shift(5)
    df['dollar_volume_pressure'] = df['amount'] / df['amount'].shift(5)
    
    df['volume_persistence'] = df['volume'].rolling(5).apply(lambda x: (x > x.shift(1)).sum(), raw=False)
    
    high_20d = df['high'].rolling(20).max()
    low_20d = df['low'].rolling(20).min()
    df['relative_price_position'] = (df['close'] - low_20d) / (high_20d - low_20d)
    
    df['regime_threshold'] = df['volume_shock_intensity'] * df['dollar_volume_pressure']
    
    # Intraday Efficiency & Reversal Context
    df['price_range_efficiency'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['intraday_strength_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['overnight_gap_persistence'] = (df['open'] / df['close'].shift(1)) - 1
    df['short_term_reversal'] = ((df['close'] / df['close'].shift(1) - 1) * 
                                (df['close'] / df['close'].shift(3) - 1))
    df['range_expansion'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    df['gap_range_interaction'] = df['overnight_gap_persistence'] * df['price_range_efficiency']
    
    # Regime-Adaptive Signal Generation
    df['volume_momentum_alignment'] = df['momentum_elasticity_div'] * df['volume_shock_intensity']
    df['gap_efficiency_reversal'] = df['gap_range_interaction'] * df['short_term_reversal'] * df['intraday_strength_ratio']
    df['momentum_volume_convergence'] = df['momentum_convergence'] * df['dollar_volume_pressure']
    
    df['divergence_based_reversal'] = df['momentum_elasticity_div'] * df['volume_persistence'] * df['range_expansion']
    df['efficiency_weighted_momentum'] = df['price_range_efficiency'] * df['momentum_convergence']
    
    df['pressure_accumulation'] = df['volume_shock_intensity'].rolling(3).sum()
    df['regime_shift_signal'] = df['pressure_accumulation'] * df['momentum_elasticity_div'] * df['short_term_reversal']
    df['transition_momentum'] = df['momentum_convergence'] * df['pressure_accumulation']
    
    # Multi-Dimensional Signal Integration
    df['core_momentum_reversal_eff'] = df['volume_momentum_alignment'] * df['gap_efficiency_reversal']
    
    high_pressure_signals = df['core_momentum_reversal_eff'] * df['regime_threshold']
    low_pressure_signals = df['divergence_based_reversal'] * df['efficiency_weighted_momentum'] * (1 - df['regime_threshold'])
    transition_signals = df['regime_shift_signal'] * df['transition_momentum'] * df['pressure_accumulation']
    
    df['regime_weighted_signals'] = np.where(df['regime_threshold'] > df['regime_threshold'].rolling(10).mean(), 
                                           high_pressure_signals, 
                                           np.where(df['pressure_accumulation'] > df['pressure_accumulation'].rolling(5).mean(),
                                                   transition_signals, low_pressure_signals))
    
    df['momentum_volume_divergence'] = df['regime_weighted_signals'] * df['dollar_volume_pressure']
    df['level_adaptive_scaling'] = df['momentum_volume_divergence'] * (1 - np.abs(df['relative_price_position'] - 0.5))
    
    df['persistence_enhancement'] = df['momentum_volume_divergence'] * df['core_momentum_reversal_eff'].rolling(5).apply(
        lambda x: (x > 0).sum(), raw=False)
    
    df['efficiency_weighted_divergence'] = df['level_adaptive_scaling'] * df['persistence_enhancement']
    
    # Risk and Volatility Context
    df['returns'] = df['close'].pct_change()
    df['recent_price_volatility'] = df['returns'].rolling(5).std()
    df['range_volatility'] = ((df['high'] - df['low']) / df['close']).rolling(5).std()
    
    df['volatility_adjusted_signals'] = df['efficiency_weighted_divergence'] / df['recent_price_volatility']
    df['range_efficiency_context'] = df['volatility_adjusted_signals'] * df['price_range_efficiency'] * df['range_expansion']
    
    # Final Alpha Generation
    df['volume_pressure_confirmation'] = (df['range_efficiency_context'] * 
                                        df['dollar_volume_pressure'] * 
                                        df['volume_persistence'])
    
    df['liquidity_persistence_adjustment'] = (df['volume'] * df['amount']).rolling(5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0], raw=False)
    
    # Apply contrarian logic for extreme positions
    extreme_pos_mask = (df['relative_price_position'] > 0.8) | (df['relative_price_position'] < 0.2)
    df['final_alpha'] = df['volume_pressure_confirmation'] * df['liquidity_persistence_adjustment']
    df.loc[extreme_pos_mask, 'final_alpha'] = -df.loc[extreme_pos_mask, 'final_alpha']
    
    return df['final_alpha']
