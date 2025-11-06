import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Regime Detection & Selection
    # Volatility Fractal Regime
    data['short_term_vol'] = (data['high'] - data['low']).rolling(window=3, min_periods=1).sum()
    data['medium_term_vol'] = (data['high'] - data['low']).rolling(window=10, min_periods=1).sum()
    data['vol_regime'] = np.where(data['short_term_vol'] > data['medium_term_vol'] / 3, 
                                 'High-Volatility', 'Low-Volatility')
    
    # Microstructure Fractal Regime
    data['price_gap_eff'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001)
    data['volume_gap_eff'] = data['volume'] / (np.abs(data['close'] - data['open']) + 0.0001)
    data['micro_regime'] = np.where(data['price_gap_eff'] > data['volume_gap_eff'], 
                                   'Efficiency-Dominant', 'Volume-Dominant')
    
    # Combined Regime Selection
    data['regime_high_eff'] = (data['vol_regime'] == 'High-Volatility') & (data['micro_regime'] == 'Efficiency-Dominant')
    data['regime_high_vol'] = (data['vol_regime'] == 'High-Volatility') & (data['micro_regime'] == 'Volume-Dominant')
    data['regime_low_eff'] = (data['vol_regime'] == 'Low-Volatility') & (data['micro_regime'] == 'Efficiency-Dominant')
    data['regime_low_vol'] = (data['vol_regime'] == 'Low-Volatility') & (data['micro_regime'] == 'Volume-Dominant')
    
    # Fractal Asymmetric Momentum Components
    # High-Volatility Asymmetric Momentum
    data['price_range_5d'] = data['high'].rolling(window=5, min_periods=1).max() - data['low'].rolling(window=5, min_periods=1).min()
    data['fractal_extreme_move'] = ((data['close'] - data['close'].shift(1)) / (data['price_range_5d'] + 0.0001)) * \
                                  (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001))
    
    data['vol_avg_5d'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean().shift(5)
    data['prev_day_mid'] = (data['high'].shift(1) + data['low'].shift(1)) / 2
    data['prev_day_range'] = (data['high'].shift(1) - data['low'].shift(1)) / 2
    data['fractal_vol_breakout'] = ((data['high'] - data['low']) / (data['vol_avg_5d'] + 0.0001)) * \
                                  ((data['open'] - data['prev_day_mid']) / (data['prev_day_range'] + 0.0001))
    
    data['high_vol_momentum'] = data['fractal_extreme_move'] * data['fractal_vol_breakout']
    
    # Low-Volatility Asymmetric Momentum
    data['close_avg_5d'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['close_range_5d'] = data['close'].rolling(window=5, min_periods=1).max() - data['close'].rolling(window=5, min_periods=1).min()
    data['fractal_steady_trend'] = ((data['close'] - data['close_avg_5d']) / (data['close_range_5d'] + 0.0001)) * \
                                  ((data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001))
    
    # Calculate consistency of price direction
    price_direction = np.sign(data['close'] - data['close'].shift(1))
    consistency_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 6:
            window_directions = price_direction.iloc[i-6:i+1]
            current_direction = price_direction.iloc[i]
            consistency_count.iloc[i] = (window_directions == current_direction).sum() / 7
        else:
            consistency_count.iloc[i] = 0.5
    
    data['fractal_consistency'] = consistency_count * (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001))
    data['low_vol_momentum'] = data['fractal_steady_trend'] * data['fractal_consistency']
    
    # Regime-Adaptive Momentum Selection
    data['regime_momentum'] = (
        data['regime_high_eff'] * data['high_vol_momentum'] * data['price_gap_eff'] +
        data['regime_high_vol'] * data['high_vol_momentum'] * data['volume_gap_eff'] +
        data['regime_low_eff'] * data['low_vol_momentum'] * data['price_gap_eff'] +
        data['regime_low_vol'] * data['low_vol_momentum'] * data['volume_gap_eff']
    )
    
    # Fractal Microstructure Pressure Analysis
    data['max_open_close'] = np.maximum(data['open'], data['close'])
    data['min_open_close'] = np.minimum(data['open'], data['close'])
    data['volume_avg_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    
    data['fractal_opening_rejection'] = ((data['high'] - data['max_open_close']) / (data['high'] - data['low'] + 0.0001)) * \
                                       (data['volume'] / (data['volume_avg_5d'] + 0.0001))
    
    data['fractal_closing_pressure'] = ((data['min_open_close'] - data['low']) / (data['high'] - data['low'] + 0.0001)) * \
                                      (data['volume'] / (data['volume_avg_5d'] + 0.0001))
    
    data['fractal_pressure_balance'] = data['fractal_closing_pressure'] - data['fractal_opening_rejection']
    data['intraday_range_3d'] = (data['high'] - data['low']).rolling(window=3, min_periods=1).sum()
    data['fractal_pressure_momentum'] = data['fractal_pressure_balance'] * \
                                       (np.abs(data['close'] - data['open']) / (data['intraday_range_3d'] + 0.0001))
    
    # Volume-Entropy Microstructure Integration
    data['volume_volatility_entropy'] = (data['volume'] / (data['volume'] - data['volume'].shift(1) + 0.0001)) * \
                                       (data['volume'] / (data['volume'].shift(1) + 0.0001))
    
    data['gap_volume_fractal'] = np.abs(data['close'] - data['open']) / (data['volume'] - data['volume'].shift(1) + 0.0001)
    data['volume_entropy_resonance'] = data['volume_volatility_entropy'] * data['gap_volume_fractal']
    data['volume_pressure_integration'] = data['volume_entropy_resonance'] * data['fractal_pressure_balance']
    
    # Amount-Volume Fractal Dynamics
    data['per_unit_impact'] = ((data['close'] - data['open']) * (data['amount'] / (data['volume'] + 0.0001)) / 
                              (data['high'] - data['low'] + 0.0001)) * data['volume_volatility_entropy']
    
    data['amount_momentum'] = data['per_unit_impact'] * (data['amount'] / (data['amount'].shift(1) + 0.0001)) * \
                             data['volume_volatility_entropy']
    
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 0.0001)
    
    # Calculate volume trend consistency
    volume_direction = np.sign(data['volume'] - data['volume'].shift(1))
    volume_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_directions = volume_direction.iloc[i-4:i+1]
            current_direction = volume_direction.iloc[i]
            volume_consistency.iloc[i] = (window_directions == current_direction).sum() / 5
        else:
            volume_consistency.iloc[i] = 0.5
    
    data['volume_trend_consistency'] = volume_consistency
    data['amount_volume_signal'] = data['amount_momentum'] * data['volume_acceleration'] * data['volume_trend_consistency']
    
    # Adaptive Fractal Alpha Synthesis
    data['core_fractal_signal'] = data['regime_momentum'] * data['volume_pressure_integration'] * data['amount_volume_signal']
    
    # Market Phase Adjustment
    data['market_phase'] = np.where(data['close'] > data['close'].shift(5), 1, 
                                   np.where(data['close'] < data['close'].shift(5), -0.5, 0))
    data['market_adjusted'] = data['core_fractal_signal'] * data['market_phase'] * \
                             (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001))
    
    # Volatility Filter
    data['recent_gap_sum'] = np.abs(data['close'] - data['open']).rolling(window=5, min_periods=1).sum()
    data['longer_gap_sum'] = np.abs(data['close'] - data['open']).rolling(window=10, min_periods=1).sum().shift(5)
    data['volatility_filter'] = (data['recent_gap_sum'] / (data['longer_gap_sum'] + 0.0001)) - 1
    
    # Final Alpha
    data['final_alpha'] = data['market_adjusted'] * (1 + data['volatility_filter']) * \
                         (data['amount'] / (data['amount'].shift(1) + 0.0001))
    
    return data['final_alpha']
