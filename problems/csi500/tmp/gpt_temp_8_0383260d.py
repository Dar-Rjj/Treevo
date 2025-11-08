import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # True Range Calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['TR'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Historical Volatility Baseline
    data['hist_vol_baseline'] = data['TR'].rolling(window=20, min_periods=1).mean()
    
    # Volatility Regime Identification
    data['high_vol_regime'] = data['TR'] > (1.5 * data['hist_vol_baseline'])
    
    # Asymmetric Pressure Dynamics
    data['up_day_pressure'] = np.where(
        data['close'] > data['open'],
        ((data['close'] - data['low']) / (data['high'] - data['low'])) * data['volume'],
        0
    )
    data['down_day_concentration'] = np.where(
        data['close'] < data['open'],
        ((data['high'] - data['close']) / (data['high'] - data['low'])) * data['volume'],
        0
    )
    data['asymmetric_pressure_score'] = data['up_day_pressure'] - data['down_day_concentration']
    
    # Multi-Timeframe Momentum Patterns
    data['short_term_mom_accel'] = ((data['close'] - data['close'].shift(2)) / data['close'].shift(2)) - \
                                  ((data['close'] - data['close'].shift(5)) / data['close'].shift(5))
    data['ma_5'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['trend_regime_momentum'] = np.where(
        data['close'] > data['ma_5'],
        data['short_term_mom_accel'],
        -data['short_term_mom_accel']
    )
    
    # Price Efficiency Analysis
    data['current_range_pos'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['prev_range_pos'] = (data['close'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['range_efficiency_momentum'] = data['current_range_pos'] - data['prev_range_pos']
    
    # Path Efficiency Analysis
    data['cumulative_movement'] = abs(data['close'] - data['close'].shift(1)).rolling(window=5, min_periods=1).sum()
    data['net_movement'] = abs(data['close'] - data['close'].shift(5))
    data['path_efficiency'] = data['net_movement'] / data['cumulative_movement']
    data['path_efficiency'] = data['path_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume Flow Dynamics
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_10']
    data['closing_volume_intensity'] = data['volume'] * abs(data['close'] - data['open']) / data['open']
    data['opening_volume_intensity'] = data['volume'] * abs(data['open'] - data['prev_close']) / data['prev_close']
    data['volume_flow_ratio'] = data['closing_volume_intensity'] / data['opening_volume_intensity']
    data['volume_flow_ratio'] = data['volume_flow_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Microstructure Pressure Analysis
    data['opening_gap_absorption'] = abs(data['open'] - data['prev_close']) / (data['high'] - data['low'])
    data['closing_pressure'] = abs(data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    data['pressure_score'] = data['opening_gap_absorption'] + data['closing_pressure']
    
    # Price-Volume Divergence Detection
    data['intraday_return'] = data['close'] / data['open'] - 1
    data['prev_intraday_return'] = data['close'].shift(1) / data['open'].shift(1) - 1
    data['price_acceleration'] = data['intraday_return'] - data['prev_intraday_return']
    
    data['volume_roc'] = data['volume'] / data['volume'].shift(1) - 1
    data['divergence_magnitude'] = abs(data['price_acceleration'] - data['volume_roc'])
    data['direction_alignment'] = np.sign(data['price_acceleration']) * np.sign(data['volume_roc'])
    
    # Momentum Regime Identification
    data['bullish_regime'] = (data['close'] > data['open']) & (data['close'] > data['prev_close'])
    data['bearish_regime'] = (data['close'] < data['open']) & (data['close'] < data['prev_close'])
    data['neutral_regime'] = ~(data['bullish_regime'] | data['bearish_regime'])
    
    # Pattern Consistency Validation
    data['range_efficiency_sign'] = np.sign(data['range_efficiency_momentum'])
    data['range_persistence'] = data['range_efficiency_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    
    data['volume_flow_sign'] = np.sign(data['volume_flow_ratio'] - 1)
    data['volume_pattern_alignment'] = data['volume_flow_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    
    data['momentum_sign'] = np.sign(data['short_term_mom_accel'])
    data['momentum_persistence'] = data['momentum_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    
    data['volume_roc_sign'] = np.sign(data['volume_roc'])
    data['volume_convergence_consistency'] = data['volume_roc_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    
    # Pattern Consistency Scaling
    consistency_measures = ['range_persistence', 'volume_pattern_alignment', 'momentum_persistence', 'volume_convergence_consistency']
    data['pattern_consistency_scaling'] = data[consistency_measures].mean(axis=1)
    
    # Regime-Adaptive Factor Construction
    # High Volatility Regime Factor
    data['base_component'] = data['asymmetric_pressure_score'] * data['trend_regime_momentum']
    data['volume_flow_validation'] = data['base_component'] * data['volume_flow_ratio']
    data['divergence_enhanced'] = data['volume_flow_validation'] * (1 + data['divergence_magnitude'])
    
    # Normal Volatility Regime Factor
    data['pure_convergence'] = data['range_efficiency_momentum'] * data['path_efficiency']
    data['efficiency_weighted'] = data['pure_convergence'] * data['volume_flow_ratio']
    data['pressure_enhanced'] = data['efficiency_weighted'] * data['asymmetric_pressure_score']
    
    # Regime Selection
    data['regime_selection'] = np.where(
        data['high_vol_regime'],
        data['divergence_enhanced'],
        data['pressure_enhanced']
    )
    
    # Momentum Regime Component
    data['momentum_regime_component'] = (data['bullish_regime'].astype(int) - data['bearish_regime'].astype(int)) * data['volume_flow_ratio']
    
    # Microstructure Multiplier
    data['microstructure_multiplier'] = data['pressure_score'] * data['volume_ratio']
    
    # Final Alpha Generation
    data['core_factor'] = data['regime_selection'] * data['momentum_regime_component']
    data['microstructure_adjustment'] = data['core_factor'] * data['microstructure_multiplier']
    data['consistency_enhanced'] = data['microstructure_adjustment'] * data['pattern_consistency_scaling']
    data['final_factor'] = data['consistency_enhanced'] * data['direction_alignment']
    
    return data['final_factor']
