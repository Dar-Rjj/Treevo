import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Efficiency-Structure Momentum Divergence Framework
    """
    data = df.copy()
    
    # Price efficiency calculation
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    
    # Multi-timeframe price efficiency momentum
    data['price_eff_3d'] = data['price_efficiency'].rolling(window=3, min_periods=3).mean()
    data['price_eff_5d'] = data['price_efficiency'].rolling(window=5, min_periods=5).mean()
    data['price_eff_10d'] = data['price_efficiency'].rolling(window=10, min_periods=10).mean()
    data['price_eff_20d'] = data['price_efficiency'].rolling(window=20, min_periods=20).mean()
    
    # Price efficiency acceleration
    data['ultra_short_eff_acc'] = data['price_eff_3d'] - data['price_eff_5d']
    data['short_term_eff_acc'] = data['price_eff_5d'] - data['price_eff_10d']
    data['medium_term_eff_acc'] = data['price_eff_10d'] - data['price_eff_20d']
    
    # Efficiency regime classification
    data['accelerating_efficiency'] = ((data['ultra_short_eff_acc'] > 0) & 
                                     (data['short_term_eff_acc'] > 0) & 
                                     (data['medium_term_eff_acc'] > 0)).astype(int)
    data['decelerating_efficiency'] = ((data['ultra_short_eff_acc'] < 0) & 
                                     (data['short_term_eff_acc'] > 0) & 
                                     (data['medium_term_eff_acc'] > 0)).astype(int)
    data['accelerating_inefficiency'] = ((data['ultra_short_eff_acc'] < 0) & 
                                       (data['short_term_eff_acc'] < 0) & 
                                       (data['medium_term_eff_acc'] < 0)).astype(int)
    data['decelerating_inefficiency'] = ((data['ultra_short_eff_acc'] > 0) & 
                                       (data['short_term_eff_acc'] < 0) & 
                                       (data['medium_term_eff_acc'] < 0)).astype(int)
    
    # Volume efficiency calculation
    data['volume_efficiency'] = data['amount'] / (data['volume'] + 0.001)
    data['vol_eff_3d'] = data['volume_efficiency'].rolling(window=3, min_periods=3).mean()
    data['vol_eff_5d'] = data['volume_efficiency'].rolling(window=5, min_periods=5).mean()
    data['vol_eff_10d'] = data['volume_efficiency'].rolling(window=10, min_periods=10).mean()
    data['vol_eff_20d'] = data['volume_efficiency'].rolling(window=20, min_periods=20).mean()
    
    # Volume efficiency acceleration
    data['ultra_short_vol_eff_acc'] = data['vol_eff_3d'] - data['vol_eff_5d']
    data['short_term_vol_eff_acc'] = data['vol_eff_5d'] - data['vol_eff_10d']
    data['medium_term_vol_eff_acc'] = data['vol_eff_10d'] - data['vol_eff_20d']
    
    # Range efficiency metrics
    data['range_utilization_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['range_eff_momentum'] = data['range_utilization_eff'].diff(periods=5)
    data['range_eff_acc'] = (data['range_utilization_eff'].rolling(window=3, min_periods=3).mean() - 
                           data['range_utilization_eff'].rolling(window=8, min_periods=8).mean())
    
    # Structure-based efficiency metrics
    data['prev_close'] = data['close'].shift(1)
    data['gap_efficiency'] = (abs(data['open'] - data['prev_close']) / (data['high'] - data['low'] + 0.001)) * data['price_efficiency']
    
    data['range_momentum_eff'] = data['range_utilization_eff'] * data['price_efficiency']
    
    data['daily_range'] = data['high'] - data['low']
    data['ma_20d_range'] = data['daily_range'].rolling(window=20, min_periods=20).mean()
    data['structure_compression_eff'] = (1 / ((data['daily_range'] / data['ma_20d_range'] + 0.001))) * data['price_efficiency']
    
    # Efficiency-structure synchronization
    data['efficiency_alignment'] = ((np.sign(data['ultra_short_eff_acc']) == np.sign(data['ultra_short_vol_eff_acc'])).astype(int) +
                                  (np.sign(data['short_term_eff_acc']) == np.sign(data['short_term_vol_eff_acc'])).astype(int) +
                                  (np.sign(data['medium_term_eff_acc']) == np.sign(data['medium_term_vol_eff_acc'])).astype(int)))
    
    data['structure_efficiency_convergence'] = data['range_utilization_eff'] * data['ultra_short_eff_acc']
    data['volume_efficiency_confirmation'] = data['ultra_short_vol_eff_acc'] * data['ultra_short_eff_acc']
    
    # Efficiency divergence patterns
    data['price_vol_divergence'] = ((data['ultra_short_eff_acc'] > 0) & (data['ultra_short_vol_eff_acc'] < 0)).astype(int) - \
                                 ((data['ultra_short_eff_acc'] < 0) & (data['ultra_short_vol_eff_acc'] > 0)).astype(int)
    
    # Volatility context
    data['volatility_20d'] = data['close'].rolling(window=20, min_periods=20).std()
    data['volatility_60d'] = data['close'].rolling(window=60, min_periods=60).std()
    data['high_volatility'] = (data['volatility_20d'] > data['volatility_60d']).astype(int)
    data['low_volatility'] = (data['volatility_20d'] < data['volatility_60d'].rolling(window=40, min_periods=40).mean()).astype(int)
    
    # Efficiency persistence
    data['efficiency_improvement_streak'] = 0
    for i in range(1, len(data)):
        if data['price_efficiency'].iloc[i] > data['price_efficiency'].iloc[i-1]:
            data['efficiency_improvement_streak'].iloc[i] = data['efficiency_improvement_streak'].iloc[i-1] + 1
    
    # Regime-based efficiency multipliers
    data['regime_multiplier'] = 1.0
    data.loc[data['accelerating_efficiency'] == 1, 'regime_multiplier'] = 1.5
    data.loc[data['decelerating_efficiency'] == 1, 'regime_multiplier'] = 0.7
    data.loc[data['accelerating_inefficiency'] == 1, 'regime_multiplier'] = -1.2
    data.loc[data['decelerating_inefficiency'] == 1, 'regime_multiplier'] = -0.8
    
    # Base efficiency momentum
    data['base_efficiency_momentum'] = (data['ultra_short_eff_acc'] * 0.4 + 
                                      data['short_term_eff_acc'] * 0.35 + 
                                      data['medium_term_eff_acc'] * 0.25)
    
    # Volume-enhanced efficiency
    data['volume_enhanced_efficiency'] = data['base_efficiency_momentum'] * (1 + data['volume_efficiency_confirmation'])
    
    # Structure support quality
    data['structure_support_quality'] = (data['range_utilization_eff'] * 0.4 + 
                                       data['gap_efficiency'] * 0.3 + 
                                       data['structure_compression_eff'] * 0.3)
    
    # Structure-optimized efficiency
    data['structure_optimized_efficiency'] = data['volume_enhanced_efficiency'] * data['structure_support_quality']
    
    # Enhanced efficiency signal
    data['enhanced_efficiency_signal'] = data['structure_optimized_efficiency'] * data['regime_multiplier']
    
    # Volume-timed efficiency prediction
    data['volume_timed_efficiency'] = data['enhanced_efficiency_signal'] * (1 + data['volume_efficiency_confirmation'])
    
    # Structure-confirmed efficiency alpha
    data['structure_confirmed_alpha'] = data['volume_timed_efficiency'] * data['structure_support_quality']
    
    # Efficiency transition bonus
    data['efficiency_regime_change'] = ((data['accelerating_efficiency'].diff() != 0) | 
                                      (data['decelerating_efficiency'].diff() != 0) | 
                                      (data['accelerating_inefficiency'].diff() != 0) | 
                                      (data['decelerating_inefficiency'].diff() != 0)).astype(int)
    data['efficiency_transition_bonus'] = 1 + (data['efficiency_regime_change'] * 0.2)
    
    # Final alpha generation
    data['final_alpha'] = data['structure_confirmed_alpha'] * data['efficiency_transition_bonus']
    
    # Clean up intermediate columns
    result = data['final_alpha'].copy()
    
    return result
