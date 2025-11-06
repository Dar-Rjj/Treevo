import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure no future data is used by only shifting backwards
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[f'{col}_lag1'] = data[col].shift(1)
        data[f'{col}_lag2'] = data[col].shift(2)
        data[f'{col}_lag3'] = data[col].shift(3)
    
    # Directional Volatility Dynamics
    # Bullish Pressure Momentum
    data['bull_pressure_current'] = np.where(
        data['close'] - data['low'] > 0,
        (data['high'] - data['close']) / (data['close'] - data['low']),
        0
    )
    data['bull_pressure_prev'] = np.where(
        data['close_lag1'] - data['low_lag1'] > 0,
        (data['high_lag1'] - data['close_lag1']) / (data['close_lag1'] - data['low_lag1']),
        0
    )
    data['bullish_pressure_momentum'] = data['bull_pressure_current'] - data['bull_pressure_prev']
    
    # Bearish Pressure Momentum
    data['bear_pressure_current'] = np.where(
        data['high'] - data['close'] > 0,
        (data['close'] - data['low']) / (data['high'] - data['close']),
        0
    )
    data['bear_pressure_prev'] = np.where(
        data['high_lag1'] - data['close_lag1'] > 0,
        (data['close_lag1'] - data['low_lag1']) / (data['high_lag1'] - data['close_lag1']),
        0
    )
    data['bearish_pressure_momentum'] = data['bear_pressure_current'] - data['bear_pressure_prev']
    
    # Volume-Weighted Volatility
    data['vol_diff'] = (data['high'] - data['close']) - (data['close'] - data['low'])
    data['volume_adjusted_pressure'] = data['vol_diff'] * data['volume'] / data['volume_lag2'].replace(0, np.nan)
    data['volume_pressure_divergence'] = np.sign(data['volume'] - data['volume_lag1']) * data['vol_diff']
    
    # Regime-Based Volatility Patterns
    # High-Low Volatility Ratio
    hl_range_current = data['high'] - data['low']
    hl_range_lag3 = data['high_lag3'] - data['low_lag3']
    data['high_low_vol_ratio'] = np.where(
        hl_range_lag3 > 0,
        (hl_range_current / hl_range_lag3) * data['vol_diff'],
        0
    )
    
    # Volatility Persistence
    volatility_persistence = []
    for i in range(len(data)):
        if i < 4:
            volatility_persistence.append(0)
            continue
            
        count = 0
        for j in range(i-4, i+1):
            if j < 4:
                continue
            hl_range_current_j = data.iloc[j]['high'] - data.iloc[j]['low']
            hl_range_prev_j = data.iloc[j-1]['high'] - data.iloc[j-1]['low']
            vol_diff_j = (data.iloc[j]['high'] - data.iloc[j]['close']) - (data.iloc[j]['close'] - data.iloc[j]['low'])
            
            if np.sign(hl_range_current_j - hl_range_prev_j) == np.sign(vol_diff_j):
                count += 1
        
        volatility_persistence.append(count / 5)
    
    data['volatility_persistence'] = volatility_persistence
    
    # Volume-Volatility Interaction
    data['volume_spike_volatility'] = hl_range_current * data['volume'] / data['volume_lag3'].replace(0, np.nan)
    data['volume_vol_alignment'] = np.sign(data['volume'] - data['volume_lag2']) * np.sign(
        hl_range_current - (data['high_lag1'] - data['low_lag1'])
    )
    
    # Gap Volatility Efficiency
    # Gap Volatility Absorption
    data['gap_vol_absorption'] = np.where(
        hl_range_current > 0,
        ((data['open'] - data['close_lag1']) / hl_range_current) * data['vol_diff'],
        0
    )
    
    # Gap Efficiency Ratio
    data['gap_efficiency_ratio'] = np.where(
        hl_range_current > 0,
        (abs(data['open'] - data['close_lag1']) / hl_range_current) * (data['volume'] / data['volume_lag1'].replace(0, np.nan)),
        0
    )
    
    # Intraday Volatility Patterns
    # Opening Volatility Momentum
    data['opening_vol_momentum'] = np.where(
        (data['open'] - data['low'] > 0) & (hl_range_current > 0),
        ((data['high'] - data['open']) / (data['open'] - data['low'])) * ((data['close'] - data['open']) / hl_range_current),
        0
    )
    
    # Closing Volatility Asymmetry
    data['closing_vol_asymmetry'] = np.where(
        hl_range_current > 0,
        ((data['close'] - (data['high'] + data['low'])/2) / hl_range_current) * data['vol_diff'],
        0
    )
    
    # Nonlinear Volatility Systems
    # Bullish Volatility Divergence
    data['bullish_vol_divergence'] = data['bullish_pressure_momentum'] * (data['volume'] / data['volume_lag1'].replace(0, np.nan) - 1)
    
    # Bearish Volatility Divergence
    data['bearish_vol_divergence'] = data['bearish_pressure_momentum'] * (data['volume'] / data['volume_lag1'].replace(0, np.nan) - 1)
    
    # Volume-Volatility Coherence
    data['volume_vol_coherence'] = hl_range_current * data['volume'] / data['volume_lag2'].replace(0, np.nan)
    
    # Efficiency-Volatility Alignment
    current_efficiency = np.where(
        hl_range_current > 0,
        abs(data['close'] - data['open']) / hl_range_current,
        0
    )
    prev_efficiency = np.where(
        data['high_lag1'] - data['low_lag1'] > 0,
        abs(data['close_lag1'] - data['open_lag1']) / (data['high_lag1'] - data['low_lag1']),
        0
    )
    data['efficiency_vol_alignment'] = np.sign(current_efficiency - prev_efficiency) * np.sign(
        hl_range_current - (data['high_lag1'] - data['low_lag1'])
    )
    
    # Composite Volatility Construction
    # Core Volatility Signals
    data['primary_vol_momentum'] = data['bullish_pressure_momentum'] * data['volume_adjusted_pressure']
    data['regime_transition_signal'] = data['high_low_vol_ratio'] * data['volume_spike_volatility']
    data['efficiency_enhancement'] = data['gap_efficiency_ratio'] * data['opening_vol_momentum']
    
    # Validation Layers
    data['volatility_coherence'] = data['volume_vol_coherence'] * data['efficiency_vol_alignment']
    data['pattern_confidence'] = data['volatility_persistence'] * data['volume_vol_alignment']
    
    # Final composite factor
    data['composite_volatility_factor'] = (
        data['primary_vol_momentum'] + 
        data['regime_transition_signal'] + 
        data['efficiency_enhancement'] + 
        data['volatility_coherence'] + 
        data['pattern_confidence']
    )
    
    # Clean up intermediate columns
    final_factor = data['composite_volatility_factor']
    
    return final_factor
