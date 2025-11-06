import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['gap_size'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['gap_sign'] = np.sign(data['open'] - data['close'].shift(1))
    data['daily_range'] = data['high'] - data['low']
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['amount_volume_ratio'] = data['amount'] / data['volume']
    
    # Fractal Gap Divergence
    # Ultra-short gap momentum
    data['ultra_short_gap_momentum'] = ((data['close'] - data['close'].shift(2)) / data['daily_range']) * data['gap_size']
    
    # Medium-term gap momentum
    data['high_7d'] = data['high'].rolling(window=7, min_periods=7).max()
    data['low_7d'] = data['low'].rolling(window=7, min_periods=7).min()
    data['medium_gap_momentum'] = ((data['close'] - data['close'].shift(7)) / (data['high_7d'] - data['low_7d'])) * data['gap_size']
    
    # Gap momentum fracture
    data['gap_momentum_fracture'] = data['ultra_short_gap_momentum'] - data['medium_gap_momentum'] * data['gap_sign']
    
    # Gap Acceleration Dynamics
    # Gap price acceleration
    data['price_accel_1'] = (data['close'] - data['close'].shift(1)) / data['daily_range']
    data['price_accel_2'] = (data['close'].shift(1) - data['close'].shift(2)) / (data['high'].shift(1) - data['low'].shift(1))
    data['gap_price_acceleration'] = (data['price_accel_1'] - data['price_accel_2']) * data['gap_size']
    
    # Gap volume acceleration
    data['vol_accel_1'] = data['volume'] / data['volume'].shift(1)
    data['vol_accel_2'] = data['volume'].shift(1) / data['volume'].shift(2)
    data['gap_volume_acceleration'] = (data['vol_accel_1'] - data['vol_accel_2']) * data['gap_size']
    
    # Gap acceleration alignment
    data['intraday_return'] = (data['close'] - data['open']) / data['daily_range']
    data['gap_acceleration_alignment'] = data['gap_price_acceleration'] * data['gap_volume_acceleration'] * data['intraday_return']
    
    # Gap Persistence Synthesis
    # Gap directional persistence
    def calculate_gap_persistence(row, data):
        if pd.isna(row.name):
            return np.nan
        current_idx = data.index.get_loc(row.name)
        if current_idx < 4:
            return np.nan
        
        count = 0
        target_sign = row['gap_sign']
        for i in range(current_idx-4, current_idx+1):
            if i >= 0 and not pd.isna(data.iloc[i]['close']) and not pd.isna(data.iloc[i-1]['close']) if i > 0 else False:
                if np.sign(data.iloc[i]['close'] - data.iloc[i-1]['close']) == target_sign:
                    count += 1
        return count * row['gap_size']
    
    data['gap_directional_persistence'] = data.apply(lambda x: calculate_gap_persistence(x, data), axis=1)
    
    # Volume-confirmed gap persistence
    data['volume_confirmed_gap_persistence'] = data['gap_directional_persistence'] * data['volume_ratio']
    
    # Consecutive gap persistence
    def calculate_consecutive_gaps(row, data):
        if pd.isna(row.name):
            return np.nan
        current_idx = data.index.get_loc(row.name)
        if current_idx == 0:
            return 1 * row['gap_size'] * row['volume_ratio']
        
        count = 1
        current_sign = row['gap_sign']
        idx = current_idx - 1
        
        while idx >= 0 and not pd.isna(data.iloc[idx]['gap_sign']) and data.iloc[idx]['gap_sign'] == current_sign:
            count += 1
            idx -= 1
        
        return count * row['gap_size'] * row['volume_ratio']
    
    data['consecutive_gap_persistence'] = data.apply(lambda x: calculate_consecutive_gaps(x, data), axis=1)
    
    # Microstructure Gap Asymmetry
    # Gap opening imbalance
    data['gap_opening_imbalance'] = ((data['open'] - data['low']) / (data['high'] - data['open']) - 1) * np.sign(data['close'] - data['open']) * data['gap_size']
    
    # Gap opening absorption
    data['gap_opening_absorption'] = (np.abs(data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))) * data['volume_ratio'] * data['gap_size']
    
    # Gap pressure efficiency
    data['avg_trade_size_ratio'] = data['amount_volume_ratio'] / data['amount_volume_ratio'].shift(1)
    data['gap_pressure_efficiency'] = data['gap_opening_imbalance'] * data['gap_opening_absorption'] * data['avg_trade_size_ratio']
    
    # Intraday Gap Flow
    # Gap volume concentration
    data['amount_3d_sum'] = data['amount'].rolling(window=3, min_periods=3).sum()
    data['gap_volume_concentration'] = (data['amount'] / data['amount_3d_sum']) * data['gap_size']
    
    # Gap trade size momentum
    data['gap_trade_size_momentum'] = (data['avg_trade_size_ratio'] - 1) * data['gap_size']
    
    # Gap microstructure flow
    data['gap_microstructure_flow'] = data['gap_volume_concentration'] * data['gap_trade_size_momentum'] * data['intraday_return']
    
    # Closing Gap Efficiency
    # Gap closing pressure
    data['gap_closing_pressure'] = ((data['close'] - data['low']) / data['daily_range'] - (data['high'] - data['close']) / data['daily_range']) * data['gap_size']
    
    # Gap session completion
    data['gap_session_completion'] = (np.abs(data['close'] - data['open']) / data['daily_range']) * data['gap_size']
    
    # Gap closing efficiency
    data['abs_return_8d'] = np.abs(data['close'] - data['close'].shift(8))
    data['volatility_7d'] = data['close'].diff().abs().rolling(window=7, min_periods=7).sum()
    data['gap_closing_efficiency'] = data['gap_closing_pressure'] * data['gap_session_completion'] * (data['abs_return_8d'] / data['volatility_7d'])
    
    # Gap Regime Switching Dynamics
    # Volume Gap Regime
    data['volume_5d_ratio'] = data['volume'] / data['volume'].shift(5)
    
    def get_volume_regime(row):
        if pd.isna(row['volume_5d_ratio']) or pd.isna(row['gap_price_acceleration']):
            return 1.0
        if row['volume_5d_ratio'] > 1.5 and row['gap_price_acceleration'] > 0.1:
            return 2.2
        elif row['volume_5d_ratio'] < 0.7 or row['gap_price_acceleration'] < -0.1:
            return 0.6
        else:
            return 1.0
    
    # Momentum Gap Regime
    data['high_5d'] = data['high'].rolling(window=5, min_periods=5).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=5).min()
    data['momentum_5d'] = np.abs(data['close'] - data['close'].shift(5)) / (data['high_5d'] - data['low_5d'])
    
    def get_momentum_regime(row):
        if pd.isna(row['momentum_5d']):
            return 1.0
        if row['momentum_5d'] > 0.8:
            return 1.8
        elif row['momentum_5d'] < 0.3:
            return 1.4
        else:
            return 1.0
    
    # Gap Regime Interaction
    def get_regime_interaction(row):
        volume_regime = get_volume_regime(row)
        momentum_regime = get_momentum_regime(row)
        
        if volume_regime == 2.2 and momentum_regime == 1.8:
            return 2.2 * row['gap_momentum_fracture'] * row['gap_size']
        elif volume_regime == 2.2 and momentum_regime == 1.4:
            return 1.4 * row['gap_pressure_efficiency'] * row['gap_size']
        elif volume_regime == 0.6 and momentum_regime == 1.8:
            return 1.8 * row['gap_microstructure_flow'] * row['gap_size']
        elif volume_regime == 0.6 and momentum_regime == 1.4:
            return 0.6 * row['gap_closing_efficiency'] * row['gap_size']
        else:
            return 1.0 * row['gap_acceleration_alignment'] * row['gap_size']
    
    data['gap_regime_interaction'] = data.apply(get_regime_interaction, axis=1)
    
    # Final Gap Alpha Synthesis
    # Core gap momentum
    data['core_gap_momentum'] = data['gap_momentum_fracture'] * data['volume_confirmed_gap_persistence']
    
    # Enhanced gap momentum
    data['enhanced_gap_momentum'] = data['core_gap_momentum'] * data['gap_acceleration_alignment']
    
    # Regime-weighted gap
    data['regime_weighted_gap'] = data['enhanced_gap_momentum'] * data['gap_regime_interaction']
    
    # Microstructure gap integration
    data['microstructure_gap_integration'] = data['regime_weighted_gap'] * data['gap_pressure_efficiency'] * data['gap_microstructure_flow']
    
    # Volume gap refinement (Final alpha)
    data['quantum_fractal_momentum'] = data['microstructure_gap_integration'] * data['gap_closing_efficiency'] * data['gap_volume_concentration']
    
    return data['quantum_fractal_momentum']
