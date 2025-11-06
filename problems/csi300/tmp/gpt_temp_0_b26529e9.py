import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price changes and ranges
    data['close_ret_3'] = data['close'] / data['close'].shift(3) - 1
    data['high_low_range'] = data['high'] - data['low']
    data['close_open_range'] = abs(data['close'] - data['open'])
    data['close_prev_close_range'] = abs(data['close'] - data['close'].shift(1))
    data['open_prev_close_range'] = abs(data['open'] - data['close'].shift(1))
    
    # Hierarchical Gap Fracture Structure
    # Using market average as proxy for index/industry (since we don't have external data)
    market_avg_close = data['close'].rolling(window=20, min_periods=1).mean()
    market_avg_ret_3 = market_avg_close / market_avg_close.shift(3) - 1
    
    # Stock vs. Index Gap Fracture
    data['gap_fracture_index'] = (data['close_ret_3'] / (market_avg_ret_3 + 1e-6)) * (data['high_low_range'] / (data['close_prev_close_range'] + 1e-6))
    
    # Stock vs. Industry Gap Fracture (using different market proxy)
    market_avg_open = data['open'].rolling(window=15, min_periods=1).mean()
    market_avg_open_ret_3 = market_avg_open / market_avg_open.shift(3) - 1
    data['gap_fracture_industry'] = (data['close_ret_3'] / (market_avg_open_ret_3 + 1e-6)) * (data['high_low_range'] / (data['open_prev_close_range'] + 1e-6))
    
    # Hierarchical Gap Fracture Divergence
    data['hierarchical_gap_divergence'] = data['gap_fracture_index'] - data['gap_fracture_industry']
    
    # Volatility-Scaled Gap Fracture
    data['gap_fracture_vol_scaled'] = data['hierarchical_gap_divergence'] * (data['high_low_range'] / (data['close_open_range'] + 1e-6))
    data['volume_weighted_gap_fracture'] = data['gap_fracture_vol_scaled'] * (data['volume'] / (data['volume'].shift(3) + 1e-6))
    
    # Hierarchical Gap Fracture Persistence
    data['gap_divergence_sign'] = np.sign(data['hierarchical_gap_divergence'])
    data['gap_fracture_dir_persistence'] = data['gap_divergence_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: sum(x.iloc[-1] == x) if len(x) > 0 else 0, raw=False
    )
    data['gap_fracture_strength_persistence'] = data['gap_fracture_dir_persistence'] * abs(data['hierarchical_gap_divergence']) / (data['high_low_range'] + 1e-6)
    
    # Multi-Timeframe Gap Regime Analysis
    # Volatility Gap Regime Dynamics
    data['high_low_range_3'] = data['high'].shift(3) - data['low'].shift(3)
    data['high_low_range_15'] = data['high'].shift(15) - data['low'].shift(15)
    
    data['short_term_gap_vol_regime'] = (data['high_low_range'] / (data['high_low_range_3'] + 1e-6)) * (
        abs(data['close'] - data['close'].shift(2)) / (data['high_low_range'] + 1e-6)
    )
    
    data['close_ret_abs_10'] = abs(data['close'] - data['close'].shift(10))
    data['close_ret_sum_10'] = abs(data['close'] - data['close'].shift(1)).rolling(window=10, min_periods=1).sum()
    
    data['long_term_gap_vol_regime'] = (data['high_low_range'] / (data['high_low_range_15'] + 1e-6)) * (
        data['close_ret_abs_10'] / (data['close_ret_sum_10'] + 1e-6)
    )
    
    data['volatility_gap_regime_shift'] = data['short_term_gap_vol_regime'] - data['long_term_gap_vol_regime'] * np.sign(data['hierarchical_gap_divergence'])
    
    # Volume Gap Regime Dynamics
    data['gap_volume_acceleration'] = (data['volume'] / (data['volume'].shift(2) + 1e-6) - 1) * np.sign(data['close'] - data['close'].shift(1))
    
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['gap_volume_regime_strength'] = (np.log(data['volume'] + 1e-6) / np.log(data['volume_ma_5'] + 1e-6)) * (
        abs((data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-6))
    )
    
    # Gap Efficiency Regime Dynamics
    data['gap_opening_efficiency'] = ((data['close'] - data['open']) / (data['high_low_range'] + 1e-6)) * (
        (data['open'] - data['close'].shift(1)) / (abs(data['open'] - data['close'].shift(2)) + 1e-6)
    )
    
    data['gap_closing_efficiency'] = ((data['high'] - data['close']) / (data['close'] - data['low'] + 1e-6)) * (
        (data['close'] - data['open']) / (data['high_low_range'] + 1e-6)
    )
    
    # Gap Momentum Structure with Fracture Integration
    # Opening Gap Momentum Fracture
    data['hierarchical_gap_opening_pressure'] = ((data['open'] - data['low']) / (data['high'] - data['open'] + 1e-6) - 1) * np.sign(data['hierarchical_gap_divergence'])
    
    data['opening_gap_momentum'] = (data['open'] - data['close'].shift(1)) * (data['close'] - data['open']) * (
        abs(data['close'].shift(1) - data['open']) / (data['high_low_range'] + 1e-6)
    )
    
    # Intraday Gap Momentum Fracture
    data['hierarchical_gap_microstructure_flow'] = data['hierarchical_gap_divergence'] * (
        (data['close'] - data['close'].shift(1)) / (data['high_low_range'] + 1e-6)
    )
    
    data['intraday_gap_momentum'] = (np.log(data['high_low_range'] + 1e-6) / np.log(abs(data['close'] - (data['high'] + data['low']) / 2) + 1e-6)) * (
        abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high_low_range'] + 1e-6)
    )
    
    # Closing Gap Momentum Fracture
    data['hierarchical_gap_closing_efficiency'] = data['hierarchical_gap_divergence'] * (data['volume'] / (data['volume'].shift(1) + 1e-6))
    
    data['closing_gap_momentum'] = (abs(data['close'] - data['open']) / (data['high_low_range'] + 1e-6)) * (
        (data['close'] - data['close'].shift(1)) - (data['close'].shift(2) - data['close'].shift(3))
    )
    
    # Dynamic Gap Regime-Shift Detection
    # Volatility Gap Regime Classification
    data['volatility_gap_regime'] = 'normal'
    data.loc[(data['short_term_gap_vol_regime'] > 1.4) & (data['long_term_gap_vol_regime'] > 1.2), 'volatility_gap_regime'] = 'high'
    data.loc[(data['short_term_gap_vol_regime'] < 0.7) & (data['long_term_gap_vol_regime'] < 0.8), 'volatility_gap_regime'] = 'low'
    
    # Volume Gap Regime Classification
    data['volume_gap_regime'] = 'normal'
    data.loc[(data['gap_volume_acceleration'] > 1.6) & (data['gap_volume_regime_strength'] > 1.3), 'volume_gap_regime'] = 'high'
    data.loc[(data['gap_volume_acceleration'] < 0.6) & (data['gap_volume_regime_strength'] < 0.7), 'volume_gap_regime'] = 'low'
    
    # Gap Efficiency Regime Classification
    data['efficiency_gap_regime'] = 'normal'
    data.loc[(data['gap_opening_efficiency'] > 0.8) & (data['gap_closing_efficiency'] > 1.2), 'efficiency_gap_regime'] = 'high'
    data.loc[(data['gap_opening_efficiency'] < 0.2) & (data['gap_closing_efficiency'] < 0.8), 'efficiency_gap_regime'] = 'low'
    
    # Final Hierarchical Fractal Gap Momentum Alpha
    # Core Gap Momentum Fracture
    data['core_gap_momentum_fracture'] = data['opening_gap_momentum'] * data['intraday_gap_momentum'] * data['closing_gap_momentum'] * data['hierarchical_gap_divergence']
    
    # Regime Multiplier Framework
    volatility_multiplier = {
        'high': 0.5,
        'low': 0.15,
        'normal': 1.0
    }
    volume_multiplier = {
        'high': 0.35,
        'low': 0.08,
        'normal': 1.0
    }
    efficiency_multiplier = {
        'high': 0.4,
        'low': 0.12,
        'normal': 1.0
    }
    
    data['vol_mult'] = data['volatility_gap_regime'].map(volatility_multiplier)
    data['vol_vol_mult'] = data['volume_gap_regime'].map(volume_multiplier)
    data['eff_mult'] = data['efficiency_gap_regime'].map(efficiency_multiplier)
    
    data['combined_regime_multiplier'] = data['vol_mult'] * data['vol_vol_mult'] * data['eff_mult']
    
    # Volume Enhancement
    data['volume_enhancement'] = data['core_gap_momentum_fracture'] * (data['volume'] / (data['volume'].shift(3) + 1e-6)) * data['gap_fracture_strength_persistence']
    
    # Final Alpha
    data['final_alpha'] = data['volume_enhancement'] * data['combined_regime_multiplier']
    
    return data['final_alpha']
