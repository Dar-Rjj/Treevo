import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Microstructure Components
    # Price-Volume Dynamics
    data['directional_divergence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['magnitude_divergence'] = ((data['close'] / data['close'].shift(1) - 1) * 
                                   (data['volume'] / data['volume'].shift(1) - 1))
    data['acceleration_divergence'] = ((data['close'] / data['close'].shift(1) - data['close'].shift(1) / data['close'].shift(2)) * 
                                      (data['volume'] / data['volume'].shift(1) - data['volume'].shift(1) / data['volume'].shift(2)))
    
    # Range-Volume Efficiency
    data['range_expansion_efficiency'] = (data['high'] - data['low']) / data['volume']
    data['volume_concentration'] = data['volume'] / (data['high'] - data['low'])
    data['price_efficiency'] = (data['high'] - data['low']) / np.abs(data['close'] - data['close'].shift(1))
    
    # Microstructure Pressure
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['closing_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_pressure_balance'] = data['opening_pressure'] - data['closing_pressure']
    
    # Regime Detection & Classification
    # Volume Regime
    data['volume_breakout'] = ((data['volume'] > data['volume'].shift(1)) & 
                              (data['volume'] > data['volume'].shift(2))).astype(int)
    data['volume_collapse'] = ((data['volume'] < data['volume'].shift(1)) & 
                              (data['volume'] < data['volume'].shift(2))).astype(int)
    
    # Volume Persistence calculation
    volume_up_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            volume_up_count.iloc[i] = (window['volume'] > window['volume'].shift(1)).sum() / 4
        else:
            volume_up_count.iloc[i] = np.nan
    data['volume_persistence'] = volume_up_count
    
    # Efficiency Regime
    data['range_efficiency_score'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_efficiency_score'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['efficiency_classification'] = ((data['price_efficiency'] < 2) & 
                                        (data['gap_efficiency_score'] < 0.3)).astype(int)
    
    # Volatility Regime
    data['true_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Volatility Structure calculation
    volatility_structure = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            count_greater = (window['true_range'] > window['true_range'].shift(1)).sum()
            volatility_structure.iloc[i] = (data['true_range'].iloc[i] / data['true_range'].iloc[i-1]) * count_greater
        else:
            volatility_structure.iloc[i] = np.nan
    data['volatility_structure'] = volatility_structure
    
    # Volatility Classification
    data['volatility_classification'] = (data['true_range'] > data['true_range'].rolling(window=4, min_periods=1).mean().shift(1)).astype(int)
    
    # Multi-Timeframe Signal Construction
    # Immediate Momentum Signals
    data['pressure_momentum'] = data['intraday_pressure_balance'] * data['directional_divergence']
    data['volume_range_momentum'] = data['volume_concentration'] * data['range_expansion_efficiency']
    data['microstructure_flow'] = data['opening_pressure'] * data['closing_pressure']
    
    # Medium-term Momentum Signals
    # Divergence Persistence calculation
    divergence_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            pos_count = (window['directional_divergence'] > 0).sum()
            neg_count = (window['directional_divergence'] < 0).sum()
            divergence_persistence.iloc[i] = pos_count - neg_count
        else:
            divergence_persistence.iloc[i] = np.nan
    data['divergence_persistence'] = divergence_persistence
    
    data['efficiency_momentum'] = data['range_efficiency_score'] * data['volume_persistence']
    data['range_volume_momentum'] = data['volume_concentration'] * data['magnitude_divergence']
    
    # Multi-Scale Integration
    data['immediate_engine'] = data['pressure_momentum'] * data['volume_range_momentum'] * data['microstructure_flow']
    data['medium_term_engine'] = data['divergence_persistence'] * data['efficiency_momentum'] * data['range_volume_momentum']
    data['multi_scale_alignment'] = data['immediate_engine'] * data['medium_term_engine']
    
    # Regime-Adaptive Enhancement
    # Regime Amplifiers
    data['high_efficiency_multiplier'] = 1 + data['volume_persistence']
    data['low_efficiency_multiplier'] = 1 - data['volume_persistence']
    data['transition_multiplier'] = 1 + np.abs(data['divergence_persistence'])
    
    # Volatility-Weighted Pressure
    data['high_volatility_pressure'] = data['opening_pressure'] * data['volatility_structure']
    data['efficient_regime_pressure'] = data['closing_pressure'] * data['price_efficiency']
    data['transition_pressure'] = data['intraday_pressure_balance'] * data['transition_multiplier']
    
    # Regime-Specific Core
    data['high_efficiency_core'] = data['high_efficiency_multiplier'] * data['acceleration_divergence']
    data['volatility_core'] = data['high_volatility_pressure'] * data['magnitude_divergence']
    data['transition_core'] = data['transition_pressure'] * data['directional_divergence']
    
    # Exhaustion & Reversal Detection
    # Volume Exhaustion Signals
    data['overbought_volume'] = ((data['volume'] > data['volume'].shift(1)) & 
                                ((data['close'] - data['low']) / (data['high'] - data['low']) > 0.8)).astype(int)
    data['oversold_volume'] = ((data['volume'] > data['volume'].shift(1)) & 
                              ((data['close'] - data['low']) / (data['high'] - data['low']) < 0.2)).astype(int)
    data['volume_exhaustion'] = ((data['volume'] < data['volume'].shift(1)) & 
                                (data['volume'] < data['volume'].shift(2))).astype(int)
    
    # Reversal Conditions
    data['overbought_condition'] = ((data['close'] - data['low']) / (data['high'] - data['low']) > 0.8).astype(int)
    data['oversold_condition'] = ((data['close'] - data['low']) / (data['high'] - data['low']) < 0.2).astype(int)
    data['volume_divergence'] = ((data['volume'] > data['volume'].shift(1)) & 
                                (data['close'] < data['close'].shift(1))).astype(int)
    
    # Exhaustion-Reversal Multiplier
    data['bullish_exhaustion'] = (data['oversold_volume'] & data['volume_exhaustion']).astype(int)
    data['bearish_exhaustion'] = (data['overbought_volume'] & data['volume_exhaustion']).astype(int)
    
    data['exhaustion_factor'] = 0
    data.loc[data['bullish_exhaustion'] == 1, 'exhaustion_factor'] = 1
    data.loc[data['bearish_exhaustion'] == 1, 'exhaustion_factor'] = -1
    data.loc[(data['overbought_condition'] == 1) & (data['volume_divergence'] == 1), 'exhaustion_factor'] = data['exhaustion_factor'] - 1
    data.loc[(data['oversold_condition'] == 1) & (data['volume_divergence'] == 1), 'exhaustion_factor'] = data['exhaustion_factor'] + 1
    
    # Final Alpha Construction
    data['core_engine'] = data['multi_scale_alignment'] * data['volatility_structure'] * data['price_efficiency']
    data['regime_weighted_enhancement'] = data['core_engine'] * (data['high_efficiency_core'] + data['volatility_core'] + data['transition_core'])
    data['momentum_confirmation'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    data['exhaustion_adjusted_alpha'] = data['regime_weighted_enhancement'] * (1 + data['exhaustion_factor'])
    data['final_alpha'] = data['exhaustion_adjusted_alpha'] * data['momentum_confirmation'] / (1 + np.abs(data['divergence_persistence']))
    
    return data['final_alpha']
