import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Decay & Acceleration Structure
    # Multi-timeframe Momentum Half-life
    data['ret_1d'] = data['close'] / data['close'].shift(1) - 1
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    
    # Avoid division by zero and extreme values
    data['ultra_short_hl'] = -np.log(2) / np.log(1 - np.clip(data['ret_1d']**2, 1e-6, 0.999))
    data['short_term_hl'] = -np.log(2) / np.log(1 - np.clip(data['ret_3d']**2, 1e-6, 0.999))
    data['half_life_accel'] = data['ultra_short_hl'] - data['short_term_hl']
    
    # Volume-Decay Coherence
    data['volume_momentum'] = (data['volume'] / data['volume'].shift(1)) * np.sign(data['ret_1d'])
    data['decay_volume_align'] = data['volume_momentum'] * data['ultra_short_hl']
    
    # Coherence Persistence
    data['volume_momentum_sign'] = np.sign(data['volume_momentum'])
    data['coherence_persistence'] = 0
    for i in range(1, len(data)):
        if data['volume_momentum_sign'].iloc[i] == data['volume_momentum_sign'].iloc[i-1]:
            data.loc[data.index[i], 'coherence_persistence'] = data['coherence_persistence'].iloc[i-1] + 1
    
    # Momentum-Pressure Integration
    data['pressure_ratio'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)
    data['pressure_enhanced_decay'] = data['ultra_short_hl'] * data['pressure_ratio']
    data['accel_pressure_align'] = data['half_life_accel'] * (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)
    data['volume_pressure_coherence'] = data['volume_momentum'] * (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)
    
    # Microstructure Efficiency & Flow Analysis
    data['range_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)
    data['opening_gap_efficiency'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-6)
    data['price_impact_efficiency'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-6)
    
    # Asymmetric Order Flow Impact
    data['upside_flow'] = np.where(data['close'] > data['open'], 
                                  (data['high'] - data['open']) / (data['amount'] + 1e-6), 0)
    data['downside_flow'] = np.where(data['close'] < data['open'], 
                                    (data['open'] - data['low']) / (data['amount'] + 1e-6), 0)
    data['flow_imbalance'] = (data['upside_flow'] - data['downside_flow']) / (data['upside_flow'] + data['downside_flow'] + 1e-6)
    
    # Volatility-Convexity Regime Detection
    data['intraday_vol'] = ((data['high'] - data['low'])**2) / (data['close']**2 + 1e-6)
    data['overnight_vol'] = ((data['open'] - data['close'].shift(1))**2) / (data['close'].shift(1)**2 + 1e-6)
    data['vol_regime_shift'] = data['intraday_vol'] / (data['overnight_vol'] + 1e-6)
    
    data['volume_spike_convexity'] = ((data['volume'] / data['volume'].shift(1))**2) * np.sign(data['volume'] - data['volume'].shift(1))
    data['volatility_convexity'] = ((data['high'] - data['low'])**2) / (data['close']**2 + 1e-6)
    data['convexity_divergence'] = data['volume_spike_convexity'] - data['volatility_convexity']
    
    # Volatility regime duration
    data['vol_quartile'] = pd.qcut(data['intraday_vol'], 4, labels=False, duplicates='drop')
    data['vol_regime_duration'] = 0
    for i in range(1, len(data)):
        if data['vol_quartile'].iloc[i] == data['vol_quartile'].iloc[i-1]:
            data.loc[data.index[i], 'vol_regime_duration'] = data['vol_regime_duration'].iloc[i-1] + 1
    
    # Breakout & Trend Confirmation System
    data['prev_range_breakout'] = (data['close'] - data['high'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-6)
    data['intraday_breakout_strength'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)
    data['overnight_gap_breakout'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-6)
    
    data['short_term_trend'] = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-6)
    data['volume_trend_support'] = (data['volume'] / data['volume'].shift(5)) * np.sign(data['short_term_trend'])
    data['pressure_trend_align'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)) * data['short_term_trend']
    
    # Composite Alpha Integration
    # Core Signal Generation
    data['momentum_pressure_core'] = data['pressure_enhanced_decay'] * data['flow_imbalance']
    data['efficiency_confirmed_breakout'] = data['intraday_breakout_strength'] * data['range_efficiency']
    data['volume_trend_alignment'] = data['volume_trend_support'] * data['pressure_trend_align']
    
    # Volatility regime adjustment
    data['volatility_adjustment'] = 1 / (1 + np.abs(data['convexity_divergence']))
    
    # Core signal combination
    data['core_signal'] = (data['momentum_pressure_core'] + 
                          data['efficiency_confirmed_breakout'] + 
                          data['volume_trend_alignment']) * data['volatility_adjustment']
    
    # Confidence Weighting
    # Multi-timeframe consistency
    data['trend_1d'] = np.sign(data['close'] - data['close'].shift(1))
    data['trend_3d'] = np.sign(data['close'] - data['close'].shift(3))
    data['trend_5d'] = np.sign(data['short_term_trend'])
    
    data['multi_timeframe_consistency'] = ((data['trend_1d'] == data['trend_3d']) & 
                                          (data['trend_3d'] == data['trend_5d'])).astype(int)
    
    # Efficiency-Flow alignment
    high_efficiency = data['range_efficiency'] > 0.3
    strong_flow = np.abs(data['flow_imbalance']) > 0.5
    data['efficiency_flow_alignment'] = (high_efficiency & strong_flow).astype(int)
    
    # Confidence score
    data['confidence_weight'] = (
        data['multi_timeframe_consistency'] * 0.4 + 
        data['efficiency_flow_alignment'] * 0.3 + 
        (data['range_efficiency'] > 0.2).astype(int) * 0.3
    )
    
    # Signal Persistence Enhancement
    data['regime_duration_multiplier'] = 1 + data['vol_regime_duration'] / 10
    data['coherence_persistence_boost'] = 1 + data['coherence_persistence'] / 5
    
    # Final Alpha
    data['final_alpha'] = (data['core_signal'] * 
                          data['confidence_weight'] * 
                          data['regime_duration_multiplier'] * 
                          data['coherence_persistence_boost'])
    
    # Clean up and return
    alpha_series = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series
