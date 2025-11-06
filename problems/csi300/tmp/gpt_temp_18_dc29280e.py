import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Scale Momentum-Efficiency Fusion
    # Momentum-Efficiency Components
    data['micro_momentum_eff'] = ((data['close'] - data['close'].shift(1)) / 
                                 (data['high'] - data['low'] + 1e-8)) * \
                                ((data['close'] - data['close'].shift(1)) / data['close'].shift(1))
    
    data['meso_momentum_eff'] = ((data['close'] - data['close'].shift(5)) / 
                                (data['high'].rolling(6).max() - data['low'].rolling(6).min() + 1e-8)) * \
                               ((data['close'] - data['close'].shift(5)) / data['close'].shift(5))
    
    data['macro_momentum_eff'] = ((data['close'] - data['close'].shift(13)) / 
                                 (data['high'].rolling(14).max() - data['low'].rolling(14).min() + 1e-8)) * \
                                ((data['close'] - data['close'].shift(13)) / data['close'].shift(13))
    
    # Momentum-Efficiency Acceleration
    data['micro_acc'] = data['micro_momentum_eff'] - data['micro_momentum_eff'].shift(1)
    data['meso_acc'] = data['meso_momentum_eff'] - data['meso_momentum_eff'].shift(2)
    data['macro_acc'] = data['macro_momentum_eff'] - data['macro_momentum_eff'].shift(5)
    
    # Acceleration Quality
    data['acc_consistency'] = np.sign(data['micro_acc']) * np.sign(data['meso_acc']) * np.sign(data['macro_acc'])
    data['acc_magnitude'] = np.abs(data['micro_acc']) * np.abs(data['meso_acc']) * np.abs(data['macro_acc'])
    
    # Acceleration Persistence
    def count_sign_persistence(series, window):
        current_sign = np.sign(series)
        persistence = 0
        for i in range(1, window + 1):
            if series.shift(i).notna().all():
                if np.sign(series.shift(i)) == current_sign:
                    persistence += 1
        return persistence
    
    data['acc_persistence'] = data['acc_consistency'].rolling(3).apply(
        lambda x: sum(np.sign(x.iloc[1:]) == np.sign(x.iloc[0])) if len(x) == 4 else 0, raw=False
    )
    
    # Volume-Amount Absorption Dynamics
    # Absorption Components
    data['volume_absorption'] = (data['volume'] / 
                                (data['volume'].shift(1).rolling(3).mean() + 1e-8)) * \
                               (data['high'] - data['low'])
    
    data['amount_absorption'] = (data['amount'] / 
                                (data['amount'].shift(1).rolling(3).mean() + 1e-8)) * \
                               (data['high'] - data['low'])
    
    data['trade_impact_eff'] = data['amount'] / (data['volume'] * (data['high'] - data['low']) + 1e-8)
    
    # Absorption-Momentum Alignment
    data['volume_momentum_align'] = np.sign(data['volume_absorption']) * np.sign(data['micro_momentum_eff'])
    data['amount_momentum_align'] = np.sign(data['amount_absorption']) * np.sign(data['micro_momentum_eff'])
    
    data['absorption_consistency'] = ((data['volume_momentum_align'] > 0).astype(int) + 
                                     (data['amount_momentum_align'] > 0).astype(int))
    
    # Absorption Quality
    data['absorption_strength'] = data['volume_absorption'] * data['amount_absorption']
    data['absorption_momentum'] = data['absorption_strength'] - data['absorption_strength'].shift(2)
    data['quality_score'] = data['absorption_consistency'] * data['trade_impact_eff']
    
    # Range Efficiency-Pressure Integration
    # Efficiency-Pressure Components
    data['opening_eff_pressure'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * \
                                  ((data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8))
    
    data['high_eff_pressure'] = ((data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * \
                               ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8))
    
    data['low_eff_pressure'] = ((data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * \
                              ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8))
    
    # Efficiency-Pressure Momentum
    data['opening_momentum'] = data['opening_eff_pressure'] - data['opening_eff_pressure'].shift(2)
    data['high_momentum'] = data['high_eff_pressure'] - data['high_eff_pressure'].shift(2)
    data['low_momentum'] = data['low_eff_pressure'] - data['low_eff_pressure'].shift(2)
    
    # Efficiency-Pressure Divergence
    data['high_low_divergence'] = data['high_eff_pressure'] - data['low_eff_pressure']
    data['opening_range_divergence'] = data['opening_eff_pressure'] - (data['high_eff_pressure'] + data['low_eff_pressure']) / 2
    data['divergence_strength'] = np.abs(data['high_low_divergence']) * np.abs(data['opening_range_divergence'])
    
    # Fractal Gap-Regime Construction
    # Multi-Scale Gap Analysis
    data['micro_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['meso_gap'] = (data['open'] - data['close'].shift(3)) / data['close'].shift(3)
    data['macro_gap'] = (data['open'] - data['close'].shift(8)) / data['close'].shift(8)
    data['fractal_gap_cascade'] = data['micro_gap'] * data['meso_gap'] * data['macro_gap']
    
    # Gap-Momentum Alignment
    data['micro_gap_momentum'] = np.sign(data['micro_gap']) * np.sign(data['micro_momentum_eff'])
    data['meso_gap_momentum'] = np.sign(data['meso_gap']) * np.sign(data['meso_momentum_eff'])
    data['macro_gap_momentum'] = np.sign(data['macro_gap']) * np.sign(data['macro_momentum_eff'])
    
    data['gap_alignment_score'] = ((data['micro_gap_momentum'] > 0).astype(int) + 
                                  (data['meso_gap_momentum'] > 0).astype(int) + 
                                  (data['macro_gap_momentum'] > 0).astype(int))
    
    # Regime Detection
    data['high_absorption_regime'] = (data['volume_absorption'] > 1) & (data['amount_absorption'] > 1)
    data['low_absorption_regime'] = (data['volume_absorption'] <= 1) | (data['amount_absorption'] <= 1)
    data['regime_transition'] = (data['volume_absorption'] - 1) * (data['amount_absorption'] - 1) * np.sign(data['fractal_gap_cascade'])
    
    # Persistence-Convergence Framework
    # Momentum Persistence
    def count_momentum_persistence(series, window):
        current_sign = np.sign(series)
        persistence = 0
        for i in range(1, window + 1):
            if series.shift(i).notna().all():
                if np.sign(series.shift(i)) == current_sign:
                    persistence += 1
        return persistence
    
    data['short_momentum_persistence'] = data['micro_momentum_eff'].rolling(3).apply(
        lambda x: sum(np.sign(x.iloc[1:]) == np.sign(x.iloc[0])) if len(x) == 3 else 0, raw=False
    )
    
    data['medium_momentum_persistence'] = data['meso_momentum_eff'].rolling(4).apply(
        lambda x: sum(np.sign(x.iloc[1:]) == np.sign(x.iloc[0])) if len(x) == 4 else 0, raw=False
    )
    
    data['long_momentum_persistence'] = data['macro_momentum_eff'].rolling(6).apply(
        lambda x: sum(np.sign(x.iloc[1:]) == np.sign(x.iloc[0])) if len(x) == 6 else 0, raw=False
    )
    
    # Convergence Patterns
    data['volatility_asymmetry'] = ((data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)) - \
                                  ((data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8))
    
    data['volume_flow_asymmetry'] = (data['amount'] * (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) - \
                                   (data['amount'] * (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8))
    
    data['signal_convergence'] = np.sign(data['volatility_asymmetry']) * np.sign(data['volume_flow_asymmetry'])
    
    # Persistence-Convergence Quality
    data['persistence_score'] = data['short_momentum_persistence'] * data['medium_momentum_persistence'] * data['long_momentum_persistence']
    data['convergence_strength'] = np.abs(data['signal_convergence']) * np.abs(data['volatility_asymmetry']) * np.abs(data['volume_flow_asymmetry'])
    data['quality_factor'] = data['persistence_score'] * data['convergence_strength']
    
    # Hierarchical Alpha Synthesis
    # Core Momentum-Efficiency
    data['acceleration_core'] = data['acc_consistency'] * data['acc_magnitude'] * data['acc_persistence']
    data['absorption_core'] = data['absorption_strength'] * data['quality_score'] * data['absorption_momentum']
    data['efficiency_core'] = data['divergence_strength'] * data['opening_momentum']
    
    # Regime-Enhanced Components
    data['gap_regime_factor'] = data['fractal_gap_cascade'] * data['gap_alignment_score']
    
    def get_regime_weight(row):
        if row['high_absorption_regime']:
            return 1
        elif row['low_absorption_regime']:
            return -1
        else:
            return row['regime_transition']
    
    data['regime_weight'] = data.apply(get_regime_weight, axis=1)
    data['regime_enhanced'] = data['gap_regime_factor'] * data['regime_weight']
    
    # Quality Integration
    data['persistence_convergence_enhanced'] = data['regime_enhanced'] * data['quality_factor']
    data['absorption_quality_integration'] = data['persistence_convergence_enhanced'] * data['trade_impact_eff']
    
    # Final Alpha
    data['base_alpha'] = data['acceleration_core'] * data['absorption_core'] * data['efficiency_core']
    data['enhanced_alpha'] = data['base_alpha'] * data['absorption_quality_integration']
    
    return data['enhanced_alpha']
