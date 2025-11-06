import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Momentum Asymmetry & Fracture Framework
    # Multi-Timeframe Momentum Structure
    data['short_term_momentum_fracture'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(4)) - 1
    data['medium_term_momentum_fracture'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(10)) - 1
    data['momentum_fracture_divergence'] = abs(data['short_term_momentum_fracture'] - data['medium_term_momentum_fracture'])
    data['multi_timeframe_momentum_divergence'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(3)) - (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(10))
    
    # Intraday Momentum Asymmetry
    data['daily_momentum_asymmetry'] = (data['close'] - data['open']) / (data['high'] - data['low']) - 0.5
    data['upper_momentum_intensity'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
    data['lower_momentum_intensity'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
    data['momentum_asymmetry'] = data['upper_momentum_intensity'] - data['lower_momentum_intensity']
    
    # Momentum-Fracture Integration
    data['momentum_acceleration'] = data['short_term_momentum_fracture'] - data['medium_term_momentum_fracture']
    data['momentum_fracture_alignment'] = data['daily_momentum_asymmetry'] * data['momentum_asymmetry']
    data['fracture_divergence'] = (data['daily_momentum_asymmetry'] * data['momentum_asymmetry']) < 0
    data['momentum_persistence_confirmation'] = (data['close'] - data['close'].shift(1)) > (data['close'].shift(1) - data['close'].shift(2))
    
    # Liquidity-Decoupling Conformation System
    # Price-Flow Liquidity Metrics
    data['movement_liquidity'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_liquidity_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['price_discovery_liquidity'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['absorption_liquidity'] = (data['amount'] / abs(data['close'] - data['open'])) - (data['amount'].shift(1) / abs(data['close'].shift(1) - data['open'].shift(1)))
    
    # Amount-Price Decoupling Patterns
    data['amount_price_alignment'] = np.sign((data['amount'] - data['amount'].shift(5)) / data['amount'].shift(5)) * np.sign((data['close'] - data['close'].shift(3)) / data['close'].shift(3))
    data['amount_price_divergence'] = np.sign((data['amount'] - data['amount'].shift(3)) / data['amount'].shift(3)) * np.sign((data['close'] - data['close'].shift(3)) / data['close'].shift(3))
    data['amount_concentration'] = data['amount'] / data['amount'].shift(1)
    data['amount_compression'] = data['amount'] / data['amount'].rolling(window=4, min_periods=1).max().shift(1)
    
    # Liquidity-Decoupling Integration
    data['strong_liquidity_conformation'] = (data['movement_liquidity'] > 0.7) & (data['amount_concentration'] > 1)
    data['strong_decoupling_signal'] = (data['amount_price_divergence'] < 0) & (data['amount_compression'] < 0.8)
    data['amount_fracture_decoupling'] = ((abs(data['momentum_acceleration']) > 1.5) | (abs((data['amount'] - data['amount'].shift(1)) / (data['amount'].shift(1) - data['amount'].shift(2))) > 1.5)) & (np.sign(data['amount_concentration']) * np.sign(data['momentum_asymmetry']) < 0)
    data['liquidity_enhanced_fracture'] = (data['momentum_fracture_divergence'] > 0.5) & data['strong_liquidity_conformation']
    
    # Regime-Based Fracture Detection
    # Momentum Momentum Structure
    momentum_expansion = []
    momentum_contraction = []
    for i in range(len(data)):
        if i >= 3:
            exp_sum = 0
            cont_sum = 0
            for j in range(4):
                idx = i - j
                if idx >= 2:
                    momentum_change = data['close'].iloc[idx] - data['close'].iloc[idx-1]
                    prev_momentum_change = data['close'].iloc[idx-1] - data['close'].iloc[idx-2]
                    exp_sum += max(0, momentum_change - prev_momentum_change)
                    cont_sum += max(0, prev_momentum_change - momentum_change)
            momentum_expansion.append(exp_sum)
            momentum_contraction.append(cont_sum)
        else:
            momentum_expansion.append(np.nan)
            momentum_contraction.append(np.nan)
    
    data['momentum_expansion'] = momentum_expansion
    data['momentum_contraction'] = momentum_contraction
    data['momentum_momentum'] = (data['momentum_expansion'] - data['momentum_contraction']) / (data['momentum_expansion'] + data['momentum_contraction'] + 1e-8)
    
    # Fracture Regime Classification
    data['upper_fracture_regime'] = (data['momentum_asymmetry'] > 0) & (data['amount_concentration'] > 1)
    data['lower_fracture_regime'] = (data['momentum_asymmetry'] < 0) & (data['amount_concentration'] < 1)
    data['mixed_fracture_regime'] = ~(data['upper_fracture_regime'] | data['lower_fracture_regime'])
    data['momentum_regime_detection'] = (data['close'] - data['close'].shift(1)) > (data['close'].shift(5) - data['close'].shift(6))
    
    # Regime-Enhanced Signals
    data['high_momentum_regime'] = (abs(data['momentum_momentum']) > 0.4) | data['momentum_regime_detection']
    data['low_momentum_regime'] = (abs(data['momentum_momentum']) <= 0.4) & ~data['momentum_regime_detection']
    data['regime_fracture_strength'] = abs(data['momentum_asymmetry']) * abs(data['amount_concentration'])
    
    # Multi-Scale Signal Construction
    # Short-term Components
    data['short_term_component_1'] = data['momentum_acceleration'] * data['amount_concentration']
    data['short_term_component_2'] = data['daily_momentum_asymmetry'] * data['movement_liquidity']
    data['short_term_component_3'] = data['gap_liquidity_efficiency'] * data['amount_price_alignment']
    data['short_term_component_4'] = data['momentum_asymmetry'] * data['amount_price_divergence']
    
    # Medium-term Components
    data['medium_term_component_1'] = data['momentum_fracture_divergence'] * data['momentum_acceleration']
    data['medium_term_component_2'] = data['momentum_persistence_confirmation'] * data['amount_compression']
    data['medium_term_component_3'] = data['multi_timeframe_momentum_divergence'] * data['momentum_asymmetry']
    data['medium_term_component_4'] = data['amount_concentration'] * data['absorption_liquidity']
    
    # Signal Validation Framework
    data['short_term_signal'] = (data['short_term_component_1'] * data['short_term_component_2'] * 
                                data['short_term_component_3'] * data['short_term_component_4'])
    data['medium_term_signal'] = (data['medium_term_component_1'] * data['medium_term_component_2'] * 
                                 data['medium_term_component_3'] * data['medium_term_component_4'])
    data['signal_alignment'] = np.sign(data['short_term_signal']) * np.sign(data['medium_term_signal'])
    
    # Adaptive Multi-Scale Alpha Generation
    # High Momentum Regime Alpha
    data['amount_confirmed_fracture'] = (data['momentum_fracture_divergence'] > 0.5) & (data['amount_concentration'] > 1.5)
    
    # Calculate rolling sum of momentum asymmetry
    data['momentum_asymmetry_sum'] = data['momentum_asymmetry'].rolling(window=3, min_periods=1).sum()
    
    data['upper_fracture_signal'] = data['amount_fracture_decoupling'] * data['upper_fracture_regime'] * data['momentum_asymmetry_sum']
    data['lower_fracture_signal'] = data['amount_fracture_decoupling'] * data['lower_fracture_regime'] * data['momentum_asymmetry_sum']
    
    data['high_momentum_alpha'] = ((data['amount_confirmed_fracture'].astype(float) + 
                                   data['upper_fracture_signal'] + 
                                   data['lower_fracture_signal']) * 
                                  data['short_term_signal'] * 
                                  data['medium_term_signal'] * 
                                  data['signal_alignment'] * 
                                  data['momentum_fracture_alignment'])
    
    # Low Momentum Regime Alpha
    data['liquidity_fracture'] = (data['momentum_fracture_divergence'] > 0.5) & data['strong_liquidity_conformation']
    data['mixed_fracture_signal'] = data['fracture_divergence'].astype(float) * data['mixed_fracture_regime'].astype(float) * data['amount_compression']
    data['decoupling_conformation_signal'] = (data['strong_decoupling_signal'].astype(float) * 
                                             data['amount_price_divergence'] * 
                                             (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2)))
    
    data['low_momentum_alpha'] = ((data['liquidity_fracture'].astype(float) + 
                                  data['mixed_fracture_signal'] + 
                                  data['decoupling_conformation_signal']) * 
                                 data['amount_price_alignment'] * 
                                 data['amount_concentration'] * 
                                 data['movement_liquidity'] * 
                                 data['absorption_liquidity'] * 
                                 data['signal_alignment'])
    
    # Composite Fracture-Liquidity Alpha
    data['high_momentum_component'] = data['high_momentum_regime'].astype(float) * data['high_momentum_alpha']
    data['low_momentum_component'] = data['low_momentum_regime'].astype(float) * data['low_momentum_alpha']
    
    data['final_alpha'] = data['high_momentum_component'] + data['low_momentum_component']
    
    return data['final_alpha']
