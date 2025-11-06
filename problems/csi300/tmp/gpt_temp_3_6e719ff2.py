import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Volatility-Efficiency Framework
    # Fractal Volatility-Efficiency Gaps
    data['volatility_weighted_gap_eff'] = ((data['open'] - data['close'].shift(1)) / 
                                          (data['high'].shift(3) - data['low'].shift(3))) * \
                                         (abs(data['close'] - data['open']) / (data['high'] - data['low']))
    
    data['intraday_recovery_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    data['gap_fill_detection'] = np.sign(data['close'] - data['open']) != np.sign(data['open'] - data['close'].shift(1))
    
    # Efficiency-Volatility Patterns
    data['price_volatility_eff'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    data['volatility_utilization'] = data['price_volatility_eff'] / (abs(data['close'] - data['open']) * (data['high'] - data['low']))
    
    data['partial_fill_ratio'] = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1))
    data['partial_fill_ratio'] = data['partial_fill_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Microstructure Asymmetry Framework
    # Volume Concentration Asymmetry
    data['volume_efficiency'] = data['volume'] / abs(data['close'] - data['open'])
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    data['volume_volatility_coupling'] = data['volume'] / (data['high'] - data['low'])
    data['volume_volatility_coupling'] = data['volume_volatility_coupling'].replace([np.inf, -np.inf], np.nan)
    
    # Volume asymmetry calculation
    hl_midpoint = (data['high'] + data['low']) / 2
    data['volume_asymmetry'] = np.where(data['close'] > hl_midpoint, data['volume'], -data['volume'])
    data['volume_asymmetry'] = data['volume_asymmetry'] / data['volume']
    
    # Microstructure Timing Patterns
    # Spread calculation
    data['spread'] = 2 * abs(data['close'] - (data['high'] + data['low']) / 2) / ((data['high'] + data['low']) / 2)
    
    # Rolling correlation for spread-volume correlation
    data['spread_volume_corr'] = data['spread'].rolling(window=5).corr(data['volume'])
    
    # Volume timing (simplified using rolling windows)
    data['price_change'] = abs(data['close'] - data['open'])
    data['volume_timing'] = (data['volume'].rolling(window=3).apply(lambda x: x.argmax()) - 
                            data['price_change'].rolling(window=3).apply(lambda x: x.argmax())) / \
                           (data['high'].shift(2) - data['low'].shift(2))
    
    data['efficiency_volume_alignment'] = data['volatility_weighted_gap_eff'] * np.sign(data['volume_asymmetry'])
    
    # Volatility-Regime Adaptive Framework
    # Volatility Structure Classification
    data['intraday_volatility'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['overnight_volatility'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['cross_volatility_regime'] = data['intraday_volatility'] / data['overnight_volatility']
    data['cross_volatility_regime'] = data['cross_volatility_regime'].replace([np.inf, -np.inf], np.nan)
    
    # Regime-Specific Patterns
    data['volatility_regime_shift'] = data['cross_volatility_regime'] / data['cross_volatility_regime'].shift(3)
    data['volume_regime_change'] = data['volume_volatility_coupling'] / data['volume_volatility_coupling'].shift(5)
    
    data['regime_clarity'] = abs(data['close']/data['close'].shift(1) - 1) * \
                            abs(data['volume']/data['volume'].shift(1) - 1) * \
                            abs((data['high']-data['low'])/(data['high'].shift(1)-data['low'].shift(1)) - 1)
    
    # Asymmetric Momentum Construction
    # Core Momentum Components
    data['gap_strength_momentum'] = ((data['close'] - data['open']) / 
                                    (data['high'].shift(1) - data['low'].shift(1))) * data['volatility_weighted_gap_eff']
    
    data['efficiency_momentum'] = data['intraday_recovery_eff'] * data['price_volatility_eff']
    
    data['volume_microstructure_momentum'] = data['volume_timing'] * data['spread_volume_corr']
    
    # Regime-Asymmetry Processing
    data['high_vol_regime_momentum'] = (data['gap_strength_momentum'] + data['efficiency_momentum'] + 
                                       data['volume_microstructure_momentum']) / 3 * data['cross_volatility_regime']
    
    data['low_vol_regime_momentum'] = (data['gap_strength_momentum'] + data['efficiency_momentum'] + 
                                      data['volume_microstructure_momentum']) / 3 * data['volatility_utilization']
    
    data['volume_asymmetry_momentum'] = (data['gap_strength_momentum'] + data['efficiency_momentum'] + 
                                        data['volume_microstructure_momentum']) / 3 * data['volume_asymmetry'].rolling(3).mean()
    
    data['efficiency_regime_momentum'] = (data['gap_strength_momentum'] + data['efficiency_momentum'] + 
                                         data['volume_microstructure_momentum']) / 3 * data['partial_fill_ratio']
    
    # Composite Alpha Generation
    # Primary Factor Components
    data['volatility_efficiency_momentum'] = data['gap_strength_momentum'] * data['efficiency_momentum']
    
    data['microstructure_asymmetry'] = data['volume_microstructure_momentum'] * data['volume_asymmetry']
    
    data['regime_adaptive_strength'] = data['regime_clarity'] * data['volatility_regime_shift']
    
    # Conditional Signal Enhancement
    high_cross_vol = data['cross_volatility_regime'] > data['cross_volatility_regime'].rolling(10).mean()
    pos_volume_asym = data['volume_asymmetry'] > 0
    low_vol = data['intraday_volatility'] < data['intraday_volatility'].rolling(10).mean()
    high_eff = data['price_volatility_eff'] > data['price_volatility_eff'].rolling(10).mean()
    strong_volume_timing = abs(data['volume_timing']) > data['volume_timing'].abs().rolling(5).mean()
    
    enhanced_vol_eff_momentum = data['volatility_efficiency_momentum'].copy()
    enhanced_vol_eff_momentum[high_cross_vol & pos_volume_asym] *= 2
    
    enhanced_micro_asym = data['microstructure_asymmetry'].copy()
    enhanced_micro_asym[low_vol & high_eff] *= 1.5
    
    enhanced_regime_strength = data['regime_adaptive_strength'].copy()
    enhanced_regime_strength[data['gap_fill_detection'] & strong_volume_timing] *= data['volume_timing']
    
    # Final Composite Factor
    base_alpha = (enhanced_vol_eff_momentum + enhanced_micro_asym + enhanced_regime_strength) / 3
    
    asymmetry_weighted_final = base_alpha * (1 + data['volume_asymmetry'])
    
    efficiency_confirmed_output = asymmetry_weighted_final * np.sign(data['efficiency_volume_alignment'])
    
    # Return the final factor series
    return efficiency_confirmed_output
