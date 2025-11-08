import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Gap momentum calculations
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_gap_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_persistence'] = data['overnight_gap'].rolling(window=3).apply(lambda x: np.mean(np.sign(x) == np.sign(x.iloc[0])) if len(x) == 3 else np.nan)
    
    # Range-bound gap assessment
    data['upper_range_dominance'] = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['lower_range_dominance'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_range_symmetry'] = np.abs(data['upper_range_dominance'] - data['lower_range_dominance'])
    
    # Volatility-weighted gap dynamics
    data['upside_volatility'] = (data['high'] - data['close']) / data['close']
    data['downside_volatility'] = (data['close'] - data['low']) / data['close']
    data['gap_volatility_sensitivity'] = data['overnight_gap'] / (data['upside_volatility'] + data['downside_volatility']).replace(0, np.nan)
    
    # Microstructure efficiency coordination
    data['flow_efficiency'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['volume_concentration'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['amount_flow_gradient'] = data['amount'].rolling(window=5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / np.mean(x) if len(x) == 5 else np.nan)
    
    # Gap-flow interaction
    data['gap_volume_alignment'] = data['overnight_gap'] * data['volume_concentration']
    data['gap_amount_coordination'] = data['overnight_gap'] * data['flow_efficiency']
    
    # Regime classification components
    data['gap_efficiency_score'] = (
        data['intraday_gap_efficiency'] * 
        (1 - data['gap_range_symmetry']) * 
        np.abs(data['gap_volatility_sensitivity'])
    )
    
    # Volume and amount confirmation
    data['volume_confirmation'] = data['volume_concentration'].rolling(window=3).mean()
    data['amount_confirmation'] = data['flow_efficiency'].rolling(window=3).mean()
    
    # Multi-day patterns
    data['gap_momentum_3d'] = data['overnight_gap'].rolling(window=3).mean()
    data['gap_consistency'] = data['overnight_gap'].rolling(window=5).apply(lambda x: np.std(x) / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 0 else np.nan)
    
    # Composite factor construction
    # Gap efficiency core
    gap_efficiency = (
        0.4 * data['gap_efficiency_score'] +
        0.3 * (1 - data['gap_consistency'].fillna(1)) +
        0.3 * np.abs(data['gap_momentum_3d'])
    )
    
    # Microstructure confirmation
    microstructure_confirmation = (
        0.4 * data['gap_volume_alignment'] +
        0.4 * data['gap_amount_coordination'] +
        0.2 * data['amount_flow_gradient'].fillna(0)
    )
    
    # Regime-adaptive weighting
    efficiency_regime = data['gap_efficiency_score'].rolling(window=5).rank(pct=True)
    volume_regime = data['volume_confirmation'].rolling(window=5).rank(pct=True)
    
    regime_weight = (
        0.6 * efficiency_regime +
        0.4 * volume_regime
    )
    
    # Final alpha factor
    alpha_factor = (
        gap_efficiency * 
        microstructure_confirmation * 
        regime_weight
    )
    
    return alpha_factor
