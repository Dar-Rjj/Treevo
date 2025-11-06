import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Compression Intensity components
    # Range compression momentum
    data['range_comp_momentum'] = ((data['high'] - data['low']) / 
                                  (data['high'].shift(5) - data['low'].shift(5))) * \
                                 (data['volume'] / data['volume'].shift(5))
    
    # Gap compression persistence
    compression_condition = (abs(data['open'] - data['close']) < 0.5 * (data['high'] - data['low']))
    compression_ratio = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['consecutive_compression'] = compression_condition.astype(int)
    for i in range(1, len(data)):
        if compression_condition.iloc[i]:
            data['consecutive_compression'].iloc[i] = data['consecutive_compression'].iloc[i-1] + 1
    
    data['gap_comp_persistence'] = data['consecutive_compression'] * compression_ratio
    
    # Volume compression elasticity
    data['volume_comp_elasticity'] = (data['volume'] / data['volume'].shift(10)) * \
                                    ((data['high'] - data['low']) / 
                                     (data['high'].shift(10) - data['low'].shift(10)))
    
    # Volatility Breakout Potential components
    # Compression breakout signal
    range_expanding = (data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1))
    breakout_signal = ((data['close'] - data['open']) / (data['high'] - data['low'])) * compression_ratio
    data['comp_breakout_signal'] = breakout_signal * range_expanding.astype(int)
    
    # Volume breakout confirmation
    breakout_direction = np.sign(data['close'] - data['open'])
    data['volume_breakout_conf'] = (data['volume'] / data['volume'].shift(5)) * \
                                  (abs(data['close'] - data['open']) / (data['high'] - data['low'])) * \
                                  breakout_direction
    
    # Intraday Volatility Structure components
    # Morning volatility dominance
    morning_vol_ratio = (data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)
    morning_volume_ratio = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['morning_vol_dominance'] = morning_vol_ratio * morning_volume_ratio
    
    # Afternoon volatility persistence
    mid_price = (data['high'] + data['low']) / 2
    afternoon_vol = (data['close'] - mid_price) / (mid_price - data['open']).replace(0, np.nan)
    afternoon_volume_conc = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    data['afternoon_vol_persistence'] = afternoon_vol * afternoon_volume_conc
    
    # Volatility regime shift
    morning_vol = (data['high'] - data['open']) / data['open']
    afternoon_vol = (data['close'] - mid_price) / mid_price
    vol_ratio = morning_vol / afternoon_vol.replace(0, np.nan)
    volume_timing_shift = data['volume'] / data['volume'].shift(1)
    data['vol_regime_shift'] = vol_ratio * volume_timing_shift
    
    # Price-Volume Volatility Coupling components
    # Volatility efficiency
    data['vol_efficiency'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) * \
                            (data['volume'] / data['amount'])
    
    # Volume-volatility elasticity
    data['vol_vol_elasticity'] = (data['volume'] / data['volume'].shift(1)) * \
                                ((data['high'] - data['low']) / 
                                 (data['high'].shift(1) - data['low'].shift(1)))
    
    # Multi-timeframe Volatility Patterns components
    # Short-term volatility momentum
    data['short_term_vol_momentum'] = ((data['high'] - data['low']) / 
                                      (data['high'].shift(3) - data['low'].shift(3))) * \
                                     (data['volume'] / data['volume'].shift(3))
    
    # Medium-term volatility cycles
    regime_persistence = (data['high'] - data['low']).rolling(window=5, min_periods=1).std() / \
                        (data['high'] - data['low']).rolling(window=10, min_periods=1).std()
    data['medium_term_vol_cycles'] = ((data['high'] - data['low']) / 
                                     (data['high'].shift(10) - data['low'].shift(10))) * \
                                    regime_persistence
    
    # Volatility convergence
    short_term_vol = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3))
    medium_term_vol = (data['high'] - data['low']) / (data['high'].shift(10) - data['low'].shift(10))
    vol_ratio_st_mt = short_term_vol / medium_term_vol.replace(0, np.nan)
    volume_convergence = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['vol_convergence'] = vol_ratio_st_mt * volume_convergence
    
    # Volatility Regime Transitions
    # Compression to expansion
    low_vol_period = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean() < \
                    (data['high'] - data['low']).rolling(window=20, min_periods=1).mean()
    high_vol_period = (data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1))
    volume_confirmation = data['volume'] > data['volume'].rolling(window=5, min_periods=1).mean()
    data['comp_to_exp'] = (low_vol_period & high_vol_period & volume_confirmation).astype(int)
    
    # Expansion to compression
    high_vol_period_prev = (data['high'].shift(1) - data['low'].shift(1)) > \
                          (data['high'].shift(5) - data['low'].shift(5)).rolling(window=5, min_periods=1).mean()
    low_vol_current = (data['high'] - data['low']) < (data['high'].shift(1) - data['low'].shift(1))
    volume_decay = data['volume'] < data['volume'].rolling(window=5, min_periods=1).mean()
    data['exp_to_comp'] = (high_vol_period_prev & low_vol_current & volume_decay).astype(int)
    
    # Combine all components with appropriate weights
    factor = (
        0.15 * data['range_comp_momentum'].fillna(0) +
        0.12 * data['gap_comp_persistence'].fillna(0) +
        0.10 * data['volume_comp_elasticity'].fillna(0) +
        0.14 * data['comp_breakout_signal'].fillna(0) +
        0.13 * data['volume_breakout_conf'].fillna(0) +
        0.08 * data['morning_vol_dominance'].fillna(0) +
        0.07 * data['afternoon_vol_persistence'].fillna(0) +
        0.06 * data['vol_regime_shift'].fillna(0) +
        0.05 * data['vol_efficiency'].fillna(0) +
        0.04 * data['vol_vol_elasticity'].fillna(0) +
        0.03 * data['short_term_vol_momentum'].fillna(0) +
        0.02 * data['medium_term_vol_cycles'].fillna(0) +
        0.01 * data['vol_convergence'].fillna(0) +
        0.03 * data['comp_to_exp'].fillna(0) +
        0.02 * data['exp_to_comp'].fillna(0)
    )
    
    return factor
