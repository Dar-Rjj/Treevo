import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Wave Momentum Components
    data['micro_wave_momentum'] = ((data['close'] - data['close'].shift(1)) / 
                                 (data['high'] - data['low'] + 1e-8) * 
                                 (data['volume'] / (data['volume'].shift(1) + 1e-8)))
    
    data['meso_wave_momentum'] = ((data['close'] - data['close'].shift(3)) / 
                                (data['high'].rolling(4).max() - data['low'].rolling(4).min() + 1e-8) * 
                                (data['volume'] / (data['volume'].shift(3) + 1e-8)))
    
    data['macro_wave_momentum'] = ((data['close'] - data['close'].shift(7)) / 
                                 (data['high'].rolling(8).max() - data['low'].rolling(8).min() + 1e-8) * 
                                 (data['volume'] / (data['volume'].shift(7) + 1e-8)))
    
    # Wave Efficiency Dynamics
    data['price_wave_efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * 
                                   (data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8))
    
    data['volume_wave_efficiency'] = (data['volume'] / (data['volume'].shift(1) + 1e-8) * 
                                    data['volume'].shift(1) / (data['volume'].shift(2) + 1e-8))
    
    data['vwap_t'] = data['amount'] / (data['volume'] + 1e-8)
    data['vwap_t_1'] = data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8)
    data['vwap_t_2'] = data['amount'].shift(2) / (data['volume'].shift(2) + 1e-8)
    data['amount_intensity'] = (data['vwap_t'] / (data['vwap_t_1'] + 1e-8) * 
                              data['vwap_t_1'] / (data['vwap_t_2'] + 1e-8))
    
    # Wave Integration
    data['micro_wave_integration'] = data['micro_wave_momentum'] * data['price_wave_efficiency']
    data['meso_wave_integration'] = data['meso_wave_momentum'] * data['volume_wave_efficiency']
    data['macro_wave_integration'] = data['macro_wave_momentum'] * data['amount_intensity']
    
    # Volatility Regime Classification
    data['true_range'] = (data['high'] - data['low']) / (data['close'].shift(1) + 1e-8)
    data['gap_component'] = abs(data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['avg_true_range'] = data['true_range'].rolling(4).mean()
    
    data['high_volatility'] = data['true_range'] > data['avg_true_range'].shift(1)
    data['low_volatility'] = data['true_range'] < data['avg_true_range'].shift(1)
    data['transition_volatility'] = data['gap_component'] > (abs(data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8))
    
    # Flow-Wave Dynamics
    data['upward_wave_flow'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['downward_wave_flow'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['net_wave_flow'] = data['upward_wave_flow'] - data['downward_wave_flow']
    
    # Flow Persistence
    data['short_term_flow'] = np.sign(data['net_wave_flow']) * np.sign(data['net_wave_flow'].shift(1))
    
    def count_positive_flow(window):
        return (window > 0).sum() - (window < 0).sum()
    
    data['medium_term_flow'] = data['net_wave_flow'].rolling(6).apply(count_positive_flow, raw=True)
    data['flow_momentum'] = data['net_wave_flow'] / (data['net_wave_flow'].shift(3) + 1e-8)
    
    # Volatility-Flow Integration
    data['volatility_scaled_flow'] = (data['high'] - data['low']) * data['net_wave_flow'] / (data['close'] + 1e-8)
    data['flow_efficiency'] = ((data['upward_wave_flow'] - data['downward_wave_flow']) / 
                             (data['upward_wave_flow'] + data['downward_wave_flow'] + 1e-8)) * data['price_wave_efficiency']
    data['volatility_flow_alignment'] = data['volatility_scaled_flow'] * data['flow_efficiency']
    
    # Regime-Adaptive Signal Construction
    data['acceleration'] = (data['micro_wave_momentum'] > data['meso_wave_momentum']) & (data['meso_wave_momentum'] > data['macro_wave_momentum'])
    data['deceleration'] = (data['micro_wave_momentum'] < data['meso_wave_momentum']) & (data['meso_wave_momentum'] < data['macro_wave_momentum'])
    data['transition_momentum'] = ~(data['acceleration'] | data['deceleration'])
    
    # Volatility-Specific Signals
    data['high_vol_signal'] = ((data['close'] / data['close'].shift(1) - 1) * 
                             (data['volume'] / (data['high'] - data['low'] + 1e-8)))
    data['low_vol_signal'] = (((data['close'] - data['open']) / (data['open'] + 1e-8)) * 
                            (data['amount'] / (data['volume'] + 1e-8)))
    data['transition_signal'] = ((data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8) * 
                               (data['volume'] / (data['volume'].shift(1) + 1e-8)))
    
    # Flow Regime Classification
    data['flow_ratio'] = (data['upward_wave_flow'] - data['downward_wave_flow']) / (data['upward_wave_flow'] + data['downward_wave_flow'] + 1e-8)
    data['sustained_buying'] = (data['flow_ratio'] > 0) & (data['upward_wave_flow'] > data['upward_wave_flow'].shift(1))
    data['sustained_selling'] = (data['flow_ratio'] < 0) & (data['downward_wave_flow'] > data['downward_wave_flow'].shift(1))
    data['flow_reversal'] = ~(data['sustained_buying'] | data['sustained_selling'])
    
    # Multi-Fractal Temporal Structure
    # Ultra-Short Dynamics
    data['gap_wave'] = (data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + 1e-8)
    data['micro_flow'] = data['net_wave_flow'] / (data['net_wave_flow'].shift(1) + 1e-8)
    data['ultra_short_factor'] = data['gap_wave'] * data['micro_flow'] * data['price_wave_efficiency']
    
    # Short-Term Dynamics
    data['volume_wave'] = data['volume'] / (data['volume'].shift(3) + 1e-8)
    data['flow_ratio_st'] = data['net_wave_flow'] / (data['net_wave_flow'].shift(7) + 1e-8)
    data['short_term_factor'] = data['volume_wave'] * data['flow_ratio_st'] * data['medium_term_flow']
    
    # Medium-Term Dynamics
    data['efficiency_wave'] = data['price_wave_efficiency'] * data['volume_wave_efficiency']
    data['volatility_wave'] = (data['high'] - data['low']) / (data['high'].rolling(15).max() - data['low'].rolling(15).min() + 1e-8)
    data['medium_term_factor'] = data['efficiency_wave'] * data['volatility_wave']
    
    # Adaptive Factor Synthesis
    # Momentum Core
    data['momentum_core'] = np.where(
        data['acceleration'],
        data['micro_wave_integration'] * data['meso_wave_integration'],
        np.where(
            data['deceleration'],
            data['meso_wave_integration'] * data['macro_wave_integration'],
            ((data['high'] - data['low']) / (data['close'] + 1e-8)) * abs(data['micro_wave_momentum'] - data['meso_wave_momentum'])
        )
    )
    
    # Volatility Core
    data['volatility_core'] = np.where(
        data['high_volatility'],
        data['volatility_flow_alignment'] * data['flow_momentum'],
        np.where(
            data['low_volatility'],
            data['price_wave_efficiency'] * data['volume_wave_efficiency'],
            data['volatility_scaled_flow'] * data['medium_term_flow']
        )
    )
    
    # Flow Core
    data['flow_core'] = np.where(
        data['sustained_buying'],
        data['flow_ratio'] * (data['upward_wave_flow'] / (data['upward_wave_flow'].shift(1) + 1e-8)),
        np.where(
            data['sustained_selling'],
            data['flow_ratio'] * (data['downward_wave_flow'] / (data['downward_wave_flow'].shift(1) + 1e-8)),
            (1 - abs(data['flow_ratio'])) * (data['volume'] / (data['volume'].shift(1) + 1e-8))
        )
    )
    
    # Signal Integration
    data['core_signal'] = (np.sign(data['close'] / data['close'].shift(1) - 1) * 
                         np.sign((data['close'] - data['open']) / (data['open'] + 1e-8)) * 
                         (data['volume'] / (data['volume'].shift(1) + 1e-8)))
    
    data['volatility_adjustment'] = data['core_signal'] / (data['true_range'] + 1e-8)
    data['enhanced_signal'] = data['volatility_adjustment'] * np.sign((data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8))
    
    # Volatility Structure Enhancement
    data['volatility_structure'] = (data['true_range'] / (data['true_range'].shift(1) + 1e-8)) * data['true_range'].rolling(5).apply(lambda x: (x > x.iloc[-1]).sum(), raw=False)
    
    data['regime_alpha'] = np.where(
        data['high_volatility'],
        data['high_vol_signal'],
        np.where(
            data['low_volatility'],
            data['low_vol_signal'],
            data['transition_signal']
        )
    )
    
    data['structure_enhanced_alpha'] = data['regime_alpha'] * data['enhanced_signal'] * data['volatility_structure']
    
    # Multi-Scale Wave Enhancement
    data['ultra_short_enhancement'] = data['momentum_core'] * data['ultra_short_factor']
    data['short_term_enhancement'] = data['volatility_core'] * data['short_term_factor']
    data['medium_term_enhancement'] = data['flow_core'] * data['medium_term_factor']
    
    # Final Alpha Construction
    data['wave_momentum_synthesis'] = data['ultra_short_enhancement'] + data['short_term_enhancement'] + data['medium_term_enhancement']
    data['regime_adaptive_weighting'] = data['wave_momentum_synthesis'] * data['structure_enhanced_alpha']
    data['final_alpha'] = data['regime_adaptive_weighting'] * (data['upward_wave_flow'] / (data['downward_wave_flow'] + 1e-8))
    
    return data['final_alpha']
