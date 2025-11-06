import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['range'] = df['high'] - df['low']
    df['close_ret_1'] = df['close'] / df['close'].shift(1) - 1
    df['close_ret_5'] = df['close'] / df['close'].shift(5) - 1
    df['amount_ret_5'] = df['amount'] / df['amount'].shift(5) - 1
    df['range_ret_1'] = df['range'] / df['range'].shift(1) - 1
    df['range_ret_5'] = df['range'] / df['range'].shift(5) - 1
    
    # Multi-Scale Fracture Dynamics
    df['short_term_fracture_flow'] = df['close_ret_1'] - df['range_ret_1']
    df['medium_term_fracture_flow'] = df['close_ret_5'] * df['amount_ret_5'] - df['range_ret_5']
    df['fracture_flow_acceleration'] = df['short_term_fracture_flow'] - df['medium_term_fracture_flow']
    
    # Asymmetric Volume-Fracture Analysis
    df['bullish_fracture_flow'] = df['volume'] * np.maximum(0, df['close'] - df['open']) / df['range']
    df['bearish_fracture_flow'] = df['volume'] * np.maximum(0, df['open'] - df['close']) / df['range']
    df['volume_fracture_imbalance'] = (df['bullish_fracture_flow'] - df['bearish_fracture_flow']) / (df['bullish_fracture_flow'] + df['bearish_fracture_flow'] + 1e-8)
    df['fracture_persistence_flow'] = (df['range'] / df['range'].shift(1)) * np.sign(df['close'] - df['close'].shift(1))
    
    # Microstructure Fracture Integration
    df['range_efficiency'] = df['volume'] / np.maximum(df['range'], 
                                                      np.abs(df['high'] - df['close'].shift(1)),
                                                      np.abs(df['low'] - df['close'].shift(1)))
    df['path_efficiency'] = np.abs(df['close'] - df['open']) / df['range']
    df['gap_fracture'] = np.abs(df['open'] - df['close'].shift(1)) / df['range'].shift(1)
    df['order_flow_imbalance'] = ((df['close'] - df['low']) / df['range']) - ((df['high'] - df['close']) / df['range'])
    
    # Fracture Regime Classification
    df['intraday_fracture_surge'] = df['range'] / df['range'].shift(1)
    
    # Calculate rolling counts
    for i in range(len(df)):
        if i >= 4:
            # Fracture Breakout Count
            breakout_count = 0
            for j in range(i-4, i+1):
                if df.iloc[j]['range'] > 1.5 * df.iloc[j-1]['range']:
                    breakout_count += 1
            df.loc[df.index[i], 'fracture_breakout_count'] = breakout_count
            
            # Contraction Persistence
            contraction_count = 0
            for j in range(i-4, i+1):
                if df.iloc[j]['range'] < df.iloc[j-1]['range']:
                    contraction_count += 1
            df.loc[df.index[i], 'contraction_persistence'] = contraction_count
            
            # Rolling averages
            df.loc[df.index[i], 'range_efficiency_avg'] = df.iloc[i-4:i+1]['range_efficiency'].mean()
    
    df['expansion_score'] = df['fracture_breakout_count'] / 4
    df['range_compression'] = df['range'] / df['range'].shift(4)
    df['contraction_score'] = df['contraction_persistence'] / 4
    
    # Regime Classification
    df['efficient_fracture_regime'] = (df['range_efficiency'] > df['range_efficiency_avg']) & (df['gap_fracture'] < 0.3)
    df['inefficient_fracture_regime'] = (df['range_efficiency'] < df['range_efficiency_avg']) | (df['gap_fracture'] > 0.7)
    df['transition_fracture_regime'] = ~df['efficient_fracture_regime'] & ~df['inefficient_fracture_regime']
    
    # Multi-Scale Signal Construction
    df['immediate_fracture_flow'] = df['fracture_flow_acceleration'] * df['volume_fracture_imbalance']
    df['fracture_momentum'] = (df['range'] / df['range'].shift(1)) * (df['expansion_score'] - df['contraction_score'])
    df['efficiency_momentum'] = df['range_efficiency'] * df['path_efficiency']
    
    # Calculate rolling sums for medium-term components
    for i in range(len(df)):
        if i >= 4:
            df.loc[df.index[i], 'weekly_fracture_regime'] = df.iloc[i-4:i+1]['expansion_score'].sum() - df.iloc[i-4:i+1]['contraction_score'].sum()
            df.loc[df.index[i], 'volume_fracture_correlation'] = (df.iloc[i-4:i+1]['volume'] * (df.iloc[i-4:i+1]['expansion_score'] - df.iloc[i-4:i+1]['contraction_score'])).sum() / df.iloc[i-4:i+1]['volume'].sum()
    
    df['weekly_fracture_regime'] = df['weekly_fracture_regime'] / 5
    df['fracture_persistence_score'] = df['fracture_persistence_flow'] * df['range_compression']
    
    df['short_term_engine'] = df['immediate_fracture_flow'] * df['fracture_momentum'] * df['efficiency_momentum']
    df['medium_term_engine'] = df['weekly_fracture_regime'] * df['volume_fracture_correlation'] * df['fracture_persistence_score']
    df['multi_scale_alignment'] = df['short_term_engine'] * df['medium_term_engine']
    
    # Asymmetric Enhancement
    df['expansion_multiplier'] = 1 + df['expansion_score']
    df['contraction_multiplier'] = 1 + df['contraction_score']
    df['transition_multiplier'] = 1 + np.abs(df['expansion_score'] - df['contraction_score'])
    
    df['expansion_flow'] = df['volume_fracture_imbalance'] * df['expansion_multiplier']
    df['contraction_flow'] = df['order_flow_imbalance'] * df['contraction_multiplier']
    df['transition_flow'] = df['path_efficiency'] * df['transition_multiplier']
    
    df['expansion_core'] = df['expansion_flow'] * df['expansion_score']
    df['contraction_core'] = df['contraction_flow'] * df['contraction_score']
    df['transition_core'] = df['transition_flow'] * (df['expansion_score'] - df['contraction_score'])
    
    # Reversal Detection & Integration
    df['overbought_fracture'] = ((df['close'] - df['low']) / df['range'] > 0.8) & (df['volume'] > df['volume'].shift(1))
    df['oversold_fracture'] = ((df['close'] - df['low']) / df['range'] < 0.2) & (df['volume'] > df['volume'].shift(1))
    df['fracture_exhaustion'] = np.abs(df['close'] - df['close'].shift(1)) < 0.5 * df['range']
    
    df['bullish_reversal'] = df['oversold_fracture'] & df['fracture_exhaustion']
    df['bearish_reversal'] = df['overbought_fracture'] & df['fracture_exhaustion']
    
    df['reversal_factor'] = 0
    df.loc[df['bullish_reversal'], 'reversal_factor'] = 1
    df.loc[df['bearish_reversal'], 'reversal_factor'] = -1
    
    # Final Alpha Construction
    df['core_engine'] = df['multi_scale_alignment'] * df['volume_fracture_imbalance'] * df['order_flow_imbalance']
    df['regime_weighted_enhancement'] = df['core_engine'] * (df['expansion_core'] + df['contraction_core'] + df['transition_core'])
    df['reversal_adjusted_alpha'] = df['regime_weighted_enhancement'] * (1 + df['reversal_factor'])
    df['final_alpha'] = df['reversal_adjusted_alpha'] / (1 + np.abs(df['expansion_score'] - df['contraction_score']))
    
    # Fill NaN values and return
    result = df['final_alpha'].fillna(0)
    return result
