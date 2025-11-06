import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Clustered Momentum Dynamics
    # Multi-Timeframe Momentum Patterns
    data['acceleration'] = (data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))
    data['medium_term_reversal'] = (data['close']/data['close'].shift(5) - 1) - (data['close'].shift(5)/data['close'].shift(10) - 1)
    data['multi_scale_momentum_div'] = (data['close']/data['close'].shift(5) - 1) - (data['close']/data['close'].shift(10) - 1)
    
    # Volatility calculations
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_ma'] = data['daily_range'].shift(1).rolling(window=5).mean()
    
    # Volatility-Regime Momentum
    high_vol_condition = data['daily_range'] > 2 * data['volatility_ma']
    low_vol_condition = data['daily_range'] < 1 * data['volatility_ma']
    
    data['high_vol_momentum'] = np.where(high_vol_condition, data['multi_scale_momentum_div'], 0)
    data['low_vol_momentum'] = np.where(low_vol_condition, data['multi_scale_momentum_div'], 0)
    
    intraday_move = abs(data['close'] - data['open'])
    data['volatility_momentum_efficiency'] = data['multi_scale_momentum_div'] * intraday_move / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume-Momentum Interactions
    data['volume_accelerated_momentum'] = data['acceleration'] * data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    data['volume_reversal_confirmation'] = (data['medium_term_reversal'] * 
                                          np.sign(data['close'] - data['close'].shift(1)) * 
                                          data['volume'] / data['volume'].shift(5).replace(0, np.nan))
    data['momentum_volume_div'] = (data['multi_scale_momentum_div'] * 
                                 data['volume'] / data['volume'].shift(1).rolling(window=5).mean())
    
    # Entropy-Based Divergence Detection
    # Price-Volume Efficiency Patterns
    data['intraday_efficiency'] = intraday_move / (data['high'] - data['low']).replace(0, np.nan)
    data['multi_day_efficiency'] = data['intraday_efficiency'].rolling(window=5).apply(
        lambda x: np.sum(x * data['volume'].loc[x.index] if len(x) == 5 else np.nan), raw=False)
    data['efficiency_momentum'] = data['intraday_efficiency'] - data['intraday_efficiency'].shift(1)
    
    # Volume Distribution Dynamics
    data['intraday_volume_skew'] = (data['volume'] / data['volume'].shift(1).replace(0, np.nan) * 
                                  np.sign(data['close'] - data['close'].shift(1)))
    data['volume_price_corr'] = (np.sign(data['close'] - data['close'].shift(1)) * 
                               data['volume'] / data['volume'].shift(1).replace(0, np.nan))
    data['multi_timeframe_volume_trend'] = ((data['volume']/data['volume'].shift(3) - 1) / 
                                          (data['volume']/data['volume'].shift(8) - 1).replace(0, np.nan))
    
    # Cross-Timeframe Divergence Measures
    data['price_div_intensity'] = (data['close']/data['close'].shift(3) - 1) - (data['close']/data['close'].shift(8) - 1)
    data['volume_div_ratio'] = ((data['volume']/data['volume'].shift(1).rolling(window=3).mean()) / 
                              (data['volume']/data['volume'].shift(1).rolling(window=8).mean()).replace(0, np.nan))
    data['efficiency_div'] = (data['intraday_efficiency'].rolling(window=3).mean() / 
                            data['intraday_efficiency'].rolling(window=8).mean().replace(0, np.nan))
    
    # Cluster Breakout & Divergence Anchoring
    # Compression Phase Detection
    data['vol_compression'] = data['daily_range'] < 0.5 * data['volatility_ma']
    data['volume_compression'] = data['volume'] < 0.7 * data['volume'].shift(1).rolling(window=5).mean()
    data['efficiency_compression'] = data['intraday_efficiency'] < 0.5 * data['intraday_efficiency'].shift(1).rolling(window=5).mean()
    
    # Divergence-Convergence Patterns
    data['momentum_efficiency_div'] = data['multi_scale_momentum_div'] - data['intraday_efficiency']
    data['volume_price_div'] = data['volume_price_corr'] - data['medium_term_reversal']
    data['cross_timeframe_align'] = data['price_div_intensity'] * data['volume_div_ratio']
    
    # Breakout Probability Assessment
    compression_all = data['vol_compression'] & data['volume_compression'] & data['efficiency_compression']
    compression_two = ((data['vol_compression'].astype(int) + data['volume_compression'].astype(int) + 
                       data['efficiency_compression'].astype(int)) >= 2)
    
    data['breakout_prob'] = np.where(compression_all & (data['volume_div_ratio'] > data['volume_div_ratio'].shift(1)), 1.0,
                                   np.where(compression_two & (abs(data['volume'] - data['volume'].shift(1).rolling(window=5).mean()) < 
                                                           0.1 * data['volume'].shift(1).rolling(window=5).mean()), 0.5, 0.2))
    
    data['breakout_direction'] = np.sign(data['multi_scale_momentum_div']) * data['breakout_prob']
    
    # Regime-Adaptive Signal Processing
    # Volatility-Entropy Regime Classification
    data['high_div_regime'] = data['price_div_intensity'] > data['efficiency_div']
    data['low_div_regime'] = data['price_div_intensity'] < data['efficiency_div']
    data['compression_regime'] = data['vol_compression'] & data['volume_compression']
    
    # Adaptive Momentum Signals
    data['high_div_signal'] = data['momentum_efficiency_div'] * data['cross_timeframe_align']
    data['low_div_signal'] = data['volume_price_div'] * data['efficiency_momentum']
    data['compression_signal'] = data['breakout_direction'] * data['multi_day_efficiency']
    
    # Regime-Adapted Base
    data['regime_adapted_base'] = np.where(data['high_div_regime'], data['high_div_signal'],
                                         np.where(data['low_div_regime'], data['low_div_signal'],
                                                np.where(data['compression_regime'], data['compression_signal'], 0)))
    
    # Composite Alpha Construction
    # Core Signal Integration
    data['div_enhanced_momentum'] = data['regime_adapted_base'] * data['volume_reversal_confirmation']
    data['efficiency_weighted_core'] = data['div_enhanced_momentum'] * (1 + abs(data['momentum_efficiency_div']))
    data['volume_anchored_signal'] = data['efficiency_weighted_core'] * data['intraday_volume_skew']
    
    # Multi-Scale Stabilization
    data['div_stability'] = data['volume_anchored_signal'] * data['efficiency_div']
    data['volume_enhancement'] = data['div_stability'] * data['multi_timeframe_volume_trend']
    data['volatility_adjusted_factor'] = data['volume_enhancement'] * (1 + data['volatility_momentum_efficiency'])
    
    # Final Alpha Assembly
    data['cross_dimensional_alpha'] = data['volatility_adjusted_factor'] * data['cross_timeframe_align']
    data['directional_alpha'] = data['cross_dimensional_alpha'] * data['breakout_direction']
    
    # Validation & Risk Framework
    # Signal Consistency Check (simplified implementation)
    alpha_signal = data['directional_alpha']
    
    # Risk-Adjustment Layer
    data['volatility_adjustment'] = alpha_signal / (data['high'] - data['low']).replace(0, np.nan)
    data['liquidity_adjustment'] = alpha_signal * data['volume'] / data['amount'].replace(0, np.nan)
    
    # Final Risk-Adjusted Alpha
    final_alpha = data['volatility_adjustment'] * data['liquidity_adjustment']
    
    return final_alpha
