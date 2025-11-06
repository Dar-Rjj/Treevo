import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic components
    data['close_prev'] = data['close'].shift(1)
    data['volume_prev'] = data['volume'].shift(1)
    data['high_low_range'] = data['high'] - data['low']
    data['pressure_diff'] = (data['high'] - data['close']) - (data['close'] - data['low'])
    
    # Asymmetric Pressure Volatility
    data['upside_pressure_vol'] = ((data['high'] - np.maximum(data['open'], data['close'])) / data['close_prev']) * data['pressure_diff']
    data['downside_pressure_vol'] = ((np.minimum(data['open'], data['close']) - data['low']) / data['close_prev']) * data['pressure_diff']
    data['pressure_vol_asymmetry'] = data['upside_pressure_vol'] / data['downside_pressure_vol']
    data['pressure_vol_asymmetry_prev'] = data['pressure_vol_asymmetry'].shift(1)
    data['volatility_pressure_change'] = np.sign(data['pressure_vol_asymmetry'] - data['pressure_vol_asymmetry_prev'])
    
    # Pressure Regime Persistence
    data['close_change'] = data['close'] - data['close_prev']
    data['close_change_prev'] = data['close_change'].shift(1)
    data['close_change_prev2'] = data['close_change'].shift(2)
    data['pressure_diff_prev'] = data['pressure_diff'].shift(1)
    data['pressure_diff_prev2'] = data['pressure_diff'].shift(2)
    
    def calc_pressure_regime_persistence(row):
        if pd.isna(row['close_change_prev2']) or pd.isna(row['pressure_diff_prev2']):
            return 0.0
        signs = []
        for i in range(3):
            if i == 0:
                sign1 = np.sign(row['close_change'] * row['pressure_diff'])
                sign2 = np.sign(row['close_change_prev'] * row['pressure_diff_prev'])
            elif i == 1:
                sign1 = np.sign(row['close_change_prev'] * row['pressure_diff_prev'])
                sign2 = np.sign(row['close_change_prev2'] * row['pressure_diff_prev2'])
            else:
                continue
            signs.append(1 if sign1 == sign2 else 0)
        return np.mean(signs) if signs else 0.0
    
    data['pressure_regime_persistence'] = data.apply(calc_pressure_regime_persistence, axis=1)
    
    # Microstructure Pressure Imbalance
    data['high_prev'] = data['high'].shift(1)
    data['low_prev'] = data['low'].shift(1)
    data['opening_pressure_imbalance'] = ((data['open'] - data['close_prev']) / (data['high_prev'] - data['low_prev'])) * data['pressure_diff']
    data['intraday_pressure_imbalance'] = ((data['close'] - data['open']) / data['high_low_range']) * data['pressure_diff']
    
    # Cumulative Pressure Imbalance (5-day)
    data['cumulative_pressure_imbalance'] = data['intraday_pressure_imbalance'].rolling(window=5, min_periods=1).sum()
    
    # Pressure Imbalance Momentum
    data['intraday_pressure_imbalance_prev'] = data['intraday_pressure_imbalance'].shift(1)
    data['pressure_imbalance_acceleration'] = data['intraday_pressure_imbalance'] - data['intraday_pressure_imbalance_prev']
    data['pressure_imbalance_reversal'] = np.sign(data['intraday_pressure_imbalance']) * np.sign(data['intraday_pressure_imbalance_prev'])
    
    # Volume-Pressure Congruence
    data['volume_pressure_direction'] = np.sign(data['close'] - data['close_prev']) * np.sign(data['volume'] - data['volume_prev']) * data['pressure_diff']
    data['amplitude_pressure_congruence'] = (np.abs(data['close'] - data['close_prev']) / data['high_low_range']) * (data['volume'] / data['volume_prev']) * data['pressure_diff']
    
    # Pressure Congruence Persistence
    data['volume_pressure_direction_prev'] = data['volume_pressure_direction'].shift(1)
    data['volume_pressure_direction_prev2'] = data['volume_pressure_direction'].shift(2)
    
    def calc_pressure_congruence_persistence(row):
        if pd.isna(row['volume_pressure_direction_prev2']):
            return 0.0
        signs = []
        for i in range(2):
            if i == 0:
                sign1 = np.sign(row['volume_pressure_direction'])
                sign2 = np.sign(row['volume_pressure_direction_prev'])
            elif i == 1:
                sign1 = np.sign(row['volume_pressure_direction_prev'])
                sign2 = np.sign(row['volume_pressure_direction_prev2'])
            signs.append(1 if sign1 == sign2 else 0)
        return np.mean(signs) if signs else 0.0
    
    data['pressure_congruence_persistence'] = data.apply(calc_pressure_congruence_persistence, axis=1)
    
    # Pressure Congruence Momentum
    data['volume_pressure_direction_prev'] = data['volume_pressure_direction'].shift(1)
    data['amplitude_pressure_congruence_prev'] = data['amplitude_pressure_congruence'].shift(1)
    data['pressure_congruence_change'] = data['volume_pressure_direction'] - data['volume_pressure_direction_prev']
    data['amplitude_pressure_momentum'] = (data['amplitude_pressure_congruence'] / data['amplitude_pressure_congruence_prev']) - 1
    
    # Regime-Pressure Efficiency
    data['bull_pressure_efficiency'] = np.where(data['close'] > data['close_prev'], 
                                              (data['close'] - data['low']) / data['high_low_range'] * data['pressure_diff'], 0)
    data['bear_pressure_efficiency'] = np.where(data['close'] < data['close_prev'], 
                                              (data['high'] - data['close']) / data['high_low_range'] * data['pressure_diff'], 0)
    data['pressure_efficiency_ratio'] = data['bull_pressure_efficiency'] / data['bear_pressure_efficiency'].replace(0, np.nan)
    data['pressure_efficiency_ratio'] = data['pressure_efficiency_ratio'].fillna(0)
    
    # Pressure Efficiency Transition
    data['pressure_efficiency_ratio_prev'] = data['pressure_efficiency_ratio'].shift(1)
    data['pressure_efficiency_regime_change'] = np.sign(data['pressure_efficiency_ratio'] - data['pressure_efficiency_ratio_prev'])
    
    # Pressure Efficiency Persistence
    data['pressure_efficiency_ratio_prev2'] = data['pressure_efficiency_ratio'].shift(2)
    
    def calc_pressure_efficiency_persistence(row):
        if pd.isna(row['pressure_efficiency_ratio_prev2']):
            return 0.0
        signs = []
        for i in range(2):
            if i == 0:
                sign1 = np.sign(row['pressure_efficiency_ratio'])
                sign2 = np.sign(row['pressure_efficiency_ratio_prev'])
            elif i == 1:
                sign1 = np.sign(row['pressure_efficiency_ratio_prev'])
                sign2 = np.sign(row['pressure_efficiency_ratio_prev2'])
            signs.append(1 if sign1 == sign2 else 0)
        return np.mean(signs) if signs else 0.0
    
    data['pressure_efficiency_persistence'] = data.apply(calc_pressure_efficiency_persistence, axis=1)
    
    # Asymmetric Pressure Momentum
    data['upside_pressure_momentum'] = ((data['close'] - np.minimum(data['open'], data['close'])) / data['close_prev']) * data['pressure_diff']
    data['downside_pressure_momentum'] = ((np.maximum(data['open'], data['close']) - data['close']) / data['close_prev']) * data['pressure_diff']
    data['net_asymmetric_pressure_momentum'] = data['upside_pressure_momentum'] - data['downside_pressure_momentum']
    
    # Volatility-Pressure Weighted Signals
    data['volatility_pressure_upside'] = data['upside_pressure_momentum'] / data['upside_pressure_vol'].replace(0, np.nan)
    data['volatility_pressure_downside'] = data['downside_pressure_momentum'] / data['downside_pressure_vol'].replace(0, np.nan)
    data['asymmetric_volatility_pressure'] = data['volatility_pressure_upside'].fillna(0) - data['volatility_pressure_downside'].fillna(0)
    
    # Core Asymmetric Pressure Factors
    data['imbalance_pressure_factor'] = data['cumulative_pressure_imbalance'] * data['pressure_vol_asymmetry']
    data['congruence_pressure_factor'] = data['amplitude_pressure_congruence'] * data['pressure_efficiency_ratio']
    data['regime_pressure_factor'] = data['net_asymmetric_pressure_momentum'] * data['pressure_regime_persistence']
    
    # Imbalance Pressure Persistence
    data['intraday_pressure_imbalance_prev2'] = data['intraday_pressure_imbalance'].shift(2)
    
    def calc_imbalance_pressure_persistence(row):
        if pd.isna(row['intraday_pressure_imbalance_prev2']):
            return 0.0
        signs = []
        for i in range(2):
            if i == 0:
                sign1 = np.sign(row['intraday_pressure_imbalance'])
                sign2 = np.sign(row['intraday_pressure_imbalance_prev'])
            elif i == 1:
                sign1 = np.sign(row['intraday_pressure_imbalance_prev'])
                sign2 = np.sign(row['intraday_pressure_imbalance_prev2'])
            signs.append(1 if sign1 == sign2 else 0)
        return np.mean(signs) if signs else 0.0
    
    data['imbalance_pressure_persistence_val'] = data.apply(calc_imbalance_pressure_persistence, axis=1)
    data['imbalance_pressure_persistence'] = data['imbalance_pressure_factor'] * data['imbalance_pressure_persistence_val']
    
    # Other persistence components
    data['congruence_pressure_persistence'] = data['congruence_pressure_factor'] * data['pressure_congruence_persistence']
    data['regime_pressure_transition'] = data['regime_pressure_factor'] * np.abs(data['pressure_regime_persistence'] - 0.5)
    
    # Multi-Timeframe Pressure-Regime Dynamics
    data['close_prev2'] = data['close'].shift(2)
    data['volume_prev2'] = data['volume'].shift(2)
    data['close_prev9'] = data['close'].shift(9)
    data['close_prev4'] = data['close'].shift(4)
    data['volume_prev9'] = data['volume'].shift(9)
    data['volume_prev4'] = data['volume'].shift(4)
    
    # Short-Term (3-day)
    data['immediate_pressure_regime_momentum'] = ((data['close'] / data['close_prev2'] - 1) * data['pressure_diff'] * data['imbalance_pressure_factor'])
    data['volume_pressure_regime_flow'] = ((data['volume'] / data['volume_prev2'] - 1) * data['pressure_diff'] * data['amplitude_pressure_congruence'])
    
    # Medium-Term (10-day)
    data['pressure_regime_transition'] = ((data['close'] / data['close_prev9'] - 1) - (data['close_prev4'] / data['close_prev9'] - 1)) * data['pressure_diff'] * data['pressure_efficiency_ratio']
    data['volume_regime_transition'] = ((data['volume'] / data['volume_prev9'] - 1) - (data['volume_prev4'] / data['volume_prev9'] - 1)) * data['pressure_diff'] * (data['volume'] / data['volume_prev'])
    
    # Cross-Scale Pressure-Regime
    data['pressure_regime_momentum_alignment'] = data['immediate_pressure_regime_momentum'] * data['pressure_efficiency_ratio'] * data['volume_pressure_regime_flow']
    data['pressure_regime_transition_alignment'] = data['pressure_regime_transition'] * data['pressure_vol_asymmetry'] * data['volume_regime_transition']
    
    # Composite Asymmetric Pressure-Regime Alpha
    data['core_pressure_regime_alpha'] = data['imbalance_pressure_persistence'] * data['congruence_pressure_persistence'] * data['regime_pressure_transition']
    data['dynamic_pressure_regime_weight'] = np.abs(data['net_asymmetric_pressure_momentum']) * (data['volume'] / data['volume_prev']) * data['pressure_diff']
    data['multi_scale_pressure_regime_factor'] = data['pressure_regime_momentum_alignment'] * data['pressure_regime_transition_alignment'] * (data['volume'] / data['volume_prev'])
    
    # Final Alpha Signal
    data['alpha'] = (data['core_pressure_regime_alpha'] * data['dynamic_pressure_regime_weight'] * 
                    data['pressure_efficiency_ratio'] * data['multi_scale_pressure_regime_factor'])
    
    # Replace infinite values and handle NaN
    data['alpha'] = data['alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return data['alpha']
