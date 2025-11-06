import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Fractured Momentum Patterns
    data['gap_momentum'] = (data['open'] / data['close'].shift(1) - 1) * (data['volume'] / data['volume'].shift(1))
    data['intraday_momentum'] = (data['close'] / data['open'] - 1) * (data['high'] - data['low']) / data['open']
    data['fracture_momentum'] = np.abs(data['gap_momentum']) - np.abs(data['intraday_momentum']) * np.sign(data['gap_momentum'] * data['intraday_momentum'])
    
    # Persistence Components
    returns = data['close'].pct_change()
    momentum_consistency = (returns.rolling(3).apply(lambda x: (x > 0).sum() if len(x) == 3 else np.nan) / 3).fillna(0)
    volume_persistence = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    data['fracture_persistence'] = data['fracture_momentum'] * momentum_consistency * volume_persistence
    
    # Pattern Synthesis
    data['fractured_trend'] = data['fracture_momentum'] * data['fracture_persistence'] * (data['close'] / data['close'].shift(1) - 1)
    data['pattern_strength'] = np.abs(data['fractured_trend']) * momentum_consistency * (data['volume'] / data['volume'].shift(1))
    
    # Volume-Pressure Dynamics
    data['buying_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'])).replace([np.inf, -np.inf], 0) * data['volume']
    data['selling_pressure'] = ((data['high'] - data['close']) / (data['high'] - data['low'])).replace([np.inf, -np.inf], 0) * data['volume']
    data['pressure_imbalance'] = (data['buying_pressure'] - data['selling_pressure']) / (data['buying_pressure'] + data['selling_pressure']).replace(0, np.nan)
    
    # Accumulation Components
    data['volume_accumulation'] = data['volume'] / data['volume'].rolling(3).mean()
    data['price_accumulation'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['accumulation_efficiency'] = data['price_accumulation'] * data['volume_accumulation'] * data['pressure_imbalance']
    
    # Dynamics Synthesis
    data['pressure_momentum'] = data['pressure_imbalance'] * data['accumulation_efficiency'] * (data['close'] / data['close'].shift(1) - 1)
    data['volume_pressure_coherence'] = np.abs(data['pressure_momentum']) * data['volume_accumulation'] * np.sign(data['pressure_imbalance'])
    
    # Range-Expansion Asymmetry
    data['true_range_expansion'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) - 1).replace([np.inf, -np.inf], 0)
    data['close_expansion'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['expansion_ratio'] = data['true_range_expansion'] / data['close_expansion'].replace(0, np.nan)
    
    # Asymmetry Components
    data['up_expansion'] = ((data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)).replace([np.inf, -np.inf], 0)
    data['down_expansion'] = ((data['open'] - data['low']) / (data['high'] - data['open']).replace(0, np.nan)).replace([np.inf, -np.inf], 0)
    data['expansion_bias'] = data['up_expansion'] - data['down_expansion']
    
    # Asymmetry Synthesis
    data['range_momentum'] = data['expansion_ratio'] * data['expansion_bias'] * (data['close'] / data['close'].shift(1) - 1)
    data['asymmetric_expansion'] = data['range_momentum'] * data['true_range_expansion'] * (data['volume'] / data['volume'].shift(1))
    
    # Core Components
    fractured_momentum_core = data['fractured_trend'] * data['pattern_strength']
    pressure_momentum_core = data['pressure_momentum'] * data['volume_pressure_coherence']
    expansion_momentum_core = data['range_momentum'] * data['asymmetric_expansion']
    
    # Regime Adaptation
    high_fracture_regime = fractured_momentum_core * np.abs(data['expansion_ratio']) * data['volume_accumulation']
    pressure_driven_regime = pressure_momentum_core * np.abs(data['pressure_imbalance']) * data['accumulation_efficiency']
    expansion_regime = expansion_momentum_core * data['true_range_expansion'] * data['expansion_bias']
    
    # Final Alpha
    regime_weighted_fusion = (high_fracture_regime + pressure_driven_regime + expansion_regime) / 3
    adaptive_alpha = regime_weighted_fusion * (data['volume'] / data['volume'].shift(1)) * (data['close'] / data['close'].shift(1) - 1)
    
    return adaptive_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
