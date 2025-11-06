import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Efficiency-Regime Dynamics
    # Intraday Efficiency Patterns
    data['efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['efficiency'] = data['efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 5-day efficiency momentum
    data['efficiency_momentum'] = data['efficiency'].rolling(window=5).mean()
    
    # 3-day efficiency autocorrelation
    data['efficiency_autocorr'] = data['efficiency'].rolling(window=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) == 3 else 0, raw=False
    ).fillna(0)
    
    # Efficiency Regimes
    data['high_efficiency'] = (data['efficiency'] > 0.5).astype(int)
    data['low_efficiency'] = (data['efficiency'] < 0.2).astype(int)
    data['efficiency_regime_strength'] = data['high_efficiency'].rolling(window=3).sum()
    
    # 2. Volume-Weighted Microstructure
    # Directional Pressure
    data['buying_pressure'] = ((2 * data['close'] - data['low'] - data['high']) / 
                              (data['high'] - data['low'])) * data['volume']
    data['buying_pressure'] = data['buying_pressure'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['selling_pressure'] = ((data['high'] + data['low'] - 2 * data['close']) / 
                               (data['high'] - data['low'])) * data['volume']
    data['selling_pressure'] = data['selling_pressure'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 3-day pressure momentum
    data['pressure_momentum'] = (data['buying_pressure'] - data['selling_pressure']).rolling(window=3).mean()
    
    # Microstructure Regimes
    median_volume_20d = data['volume'].rolling(window=20).median()
    data['strong_microstructure'] = (data['volume'] > median_volume_20d).astype(int)
    data['weak_microstructure'] = (data['volume'] < median_volume_20d).astype(int)
    data['microstructure_persistence'] = data['strong_microstructure'].rolling(window=3).sum()
    
    # 3. Regime-Constrained Signals
    # Combine Efficiency and Microstructure Regimes
    data['regime_alignment'] = 0
    # High efficiency + Strong microstructure: momentum enhancement
    data.loc[(data['high_efficiency'] == 1) & (data['strong_microstructure'] == 1), 'regime_alignment'] = 1.5
    # Low efficiency + Weak microstructure: reversal emphasis  
    data.loc[(data['low_efficiency'] == 1) & (data['weak_microstructure'] == 1), 'regime_alignment'] = -1.2
    # Mixed regimes: signal reduction
    data.loc[((data['high_efficiency'] == 1) & (data['weak_microstructure'] == 1)) | 
             ((data['low_efficiency'] == 1) & (data['strong_microstructure'] == 1)), 'regime_alignment'] = 0.3
    
    # Calculate regime alignment strength
    data['regime_strength'] = (data['efficiency_regime_strength'] + data['microstructure_persistence']) / 6
    
    # Range and Turnover Adjustment
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    
    # Range component
    data['range_component'] = (data['high'] - data['low']) / data['atr_5d']
    data['range_component'] = data['range_component'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Turnover component
    data['dollar_turnover'] = data['volume'] * data['close']
    median_dollar_turnover_20d = data['dollar_turnover'].rolling(window=20).median()
    data['turnover_component'] = np.log(data['dollar_turnover'] / median_dollar_turnover_20d)
    data['turnover_component'] = data['turnover_component'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Multiply regime signal by range and turnover measures
    data['regime_signal'] = (data['regime_alignment'] * data['regime_strength'] * 
                            data['range_component'] * data['turnover_component'])
    
    # 4. Enhanced Alpha
    # Temporal Enhancement
    data['signal_sign'] = np.sign(data['regime_signal'])
    data['signal_sign'] = data['signal_sign'].replace(0, 1)  # Handle zeros
    data['temporal_factor'] = data['signal_sign'].rolling(window=3).apply(
        lambda x: x.prod() if len(x) == 3 else 1, raw=False
    ).fillna(1)
    
    # Multiply regime-constrained signal by temporal factor
    data['enhanced_signal'] = data['regime_signal'] * data['temporal_factor']
    
    # Amount confirmation
    median_amount_10d = data['amount'].rolling(window=10).median()
    data['amount_confirmation'] = data['amount'] / median_amount_10d
    data['amount_confirmation'] = data['amount_confirmation'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Final signal
    data['alpha_factor'] = data['enhanced_signal'] * data['amount_confirmation']
    
    return data['alpha_factor']
