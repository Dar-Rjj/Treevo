import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Efficiency Framework
    # Micro Efficiency
    data['prev_close'] = data['close'].shift(1)
    data['open_close_max'] = data[['open', 'prev_close']].max(axis=1)
    data['micro_eff'] = (data['high'] - data['open_close_max']) / (data['high'] - data['low'])
    data['micro_eff'] = data['micro_eff'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Meso Efficiency
    data['close_diff_3'] = (data['close'] - data['close'].shift(3)).abs()
    data['range_sum_3'] = (data['high'] - data['low']).rolling(window=3, min_periods=1).sum()
    data['meso_eff'] = data['close_diff_3'] / data['range_sum_3']
    data['meso_eff'] = data['meso_eff'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Macro Efficiency
    data['close_diff_5'] = (data['close'] - data['close'].shift(5)).abs()
    data['range_sum_5'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).sum()
    data['macro_eff'] = data['close_diff_5'] / data['range_sum_5']
    data['macro_eff'] = data['macro_eff'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Efficiency Momentum
    data['eff_momentum'] = (data['meso_eff'] / data['macro_eff']) - 1
    data['eff_momentum'] = data['eff_momentum'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume-Amount Confirmation System
    # Volume Momentum Quality
    data['close_gt_open'] = data['close'] > data['open']
    data['close_lt_open'] = data['close'] < data['open']
    
    # Volume Asymmetry
    up_volume_avg = data['volume'].where(data['close_gt_open']).rolling(window=3, min_periods=1).mean()
    down_volume_avg = data['volume'].where(data['close_lt_open']).rolling(window=3, min_periods=1).mean()
    data['volume_asymmetry'] = (data['volume'] / up_volume_avg) - (data['volume'] / down_volume_avg)
    data['volume_asymmetry'] = data['volume_asymmetry'].fillna(0)
    
    # Volume Acceleration
    data['volume_accel'] = ((data['volume'] / data['volume'].shift(3)) - 1) - ((data['volume'] / data['volume'].shift(5)) - 1)
    data['volume_accel'] = data['volume_accel'].fillna(0)
    
    # Volume Quality
    data['volume_quality'] = data['volume_asymmetry'] * (1 + data['volume_accel'])
    
    # Amount Momentum Quality
    data['amount_avg_3'] = data['amount'].rolling(window=3, min_periods=1).mean()
    data['amount_momentum'] = (data['amount'] / data['amount_avg_3']) - 1
    
    # Amount-Volume Alignment
    data['amount_volume_align'] = np.sign(data['amount_momentum']) * np.sign(data['volume_accel'])
    
    # Amount Quality
    data['amount_quality'] = data['amount_momentum'] * (1 + data['amount_volume_align'])
    
    # Combined Confirmation Strength
    data['volume_amount_div'] = np.sign(data['volume_accel']) * -np.sign(data['amount_momentum'])
    data['confirmation_strength'] = data['volume_quality'] * data['amount_quality']
    
    # Volatility-Regime Adaptive Framework
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = (data['high'] - data['close'].shift(1)).abs()
    data['tr3'] = (data['low'] - data['close'].shift(1)).abs()
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-Timeframe Volatility Assessment
    data['short_vol'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    data['medium_vol'] = data['true_range'].rolling(window=10, min_periods=1).mean()
    data['vol_momentum'] = (data['short_vol'] / data['medium_vol']) - 1
    
    # Volatility Regime Classification and Multipliers
    conditions = [
        data['vol_momentum'] > 0.1,
        data['vol_momentum'] < -0.1
    ]
    choices = [
        1 + data['volume_quality'],  # High Volatility
        1 + data['amount_quality']   # Low Volatility
    ]
    data['regime_multiplier'] = np.select(conditions, choices, default=1 + (data['volume_quality'] + data['amount_quality'])/2)
    
    # Price Momentum Quality Enhancement
    # Directional Consistency
    data['daily_return'] = data['close'].pct_change()
    data['return_sign'] = np.sign(data['daily_return'])
    data['sign_consistency'] = data['return_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0
    )
    
    # Momentum Stability
    data['return_5d'] = data['close'].pct_change(5)
    data['return_var_5d'] = data['daily_return'].rolling(window=5, min_periods=1).var()
    data['momentum_stability'] = data['return_5d'].abs() / (data['return_var_5d'] + 0.0001)
    data['momentum_stability'] = data['momentum_stability'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Momentum Quality
    data['momentum_quality'] = data['sign_consistency'] * data['momentum_stability']
    
    # Intraday Microstructure Integration
    # Opening Session Quality
    data['gap'] = data['open'] - data['close'].shift(1)
    data['gap_absorption'] = (data['close'] - data['open']) / data['gap'].abs()
    data['gap_absorption'] = data['gap_absorption'].replace([np.inf, -np.inf], 0).fillna(0)
    data['opening_strength'] = data['gap_absorption'] * data['micro_eff']
    
    # Closing Session Quality
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_strength'] = data['intraday_strength'].replace([np.inf, -np.inf], 0).fillna(0)
    data['closing_momentum'] = data['intraday_strength'].abs() * data['volume']
    
    # Session Alignment
    data['session_alignment'] = np.sign(data['opening_strength']) * np.sign(data['closing_momentum'])
    
    # Composite Alpha Construction
    # Core Efficiency Component
    data['base_eff_factor'] = data['eff_momentum'] * data['momentum_quality']
    data['volume_amount_enhanced'] = data['base_eff_factor'] * (1 + data['confirmation_strength'])
    
    # Regime-Adaptive Weighting
    data['regime_weighted'] = data['volume_amount_enhanced'] * data['regime_multiplier']
    
    # Microstructure Refinement
    data['session_aligned'] = data['regime_weighted'] * (1 + data['session_alignment'])
    data['quality_adjusted'] = data['session_aligned'] * (1 + data['opening_strength'] * data['closing_momentum'])
    
    # Final Alpha Output
    data['raw_alpha'] = data['quality_adjusted']
    data['final_factor'] = data['raw_alpha'] / data['short_vol']
    data['final_factor'] = data['final_factor'].replace([np.inf, -np.inf], 0).fillna(0)
    
    return data['final_factor']
