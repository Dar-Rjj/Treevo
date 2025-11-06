import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    # Intraday Volatility Regime
    data['intraday_vol_regime'] = (data['high'] - data['low']) / data['close']
    
    # Overnight Volatility Regime
    data['overnight_vol_regime'] = (data['open'] / data['close'].shift(1) - 1).abs()
    
    # Regime Persistence (volatility regime autocorrelation over 3 days)
    data['vol_regime_persistence'] = data['intraday_vol_regime'].rolling(window=3).apply(
        lambda x: x.autocorr() if len(x) == 3 and not x.isna().any() else 0, raw=False
    )
    
    # Volume Flow Dynamics
    # Volume Acceleration
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Volume Regime Persistence
    data['volume_regime_persistence'] = data['volume_acceleration'].rolling(window=3).apply(
        lambda x: x.autocorr() if len(x) == 3 and not x.isna().any() else 0, raw=False
    )
    
    # Volume-Volatility Coupling
    data['volume_volatility_coupling'] = data['volume_acceleration'] * data['intraday_vol_regime']
    
    # Price Gap Efficiency
    # Opening Gap Efficiency
    data['opening_gap_efficiency'] = ((data['open'] / data['close'].shift(1) - 1) / 
                                    (data['high'].shift(1) - data['low'].shift(1) + 1e-6))
    
    # Closing Gap Efficiency
    data['closing_gap_efficiency'] = ((data['close'] / data['open'] - 1) / 
                                    (data['high'] - data['low'] + 1e-6))
    
    # Gap Asymmetry
    data['gap_asymmetry'] = data['opening_gap_efficiency'] - data['closing_gap_efficiency']
    
    # Regime Transition Signals
    # Volatility Regime Break
    data['volatility_regime_break'] = (data['intraday_vol_regime'] - 
                                     data['intraday_vol_regime'].rolling(window=5).mean())
    
    # Volume Regime Break
    data['volume_regime_break'] = (data['volume_acceleration'] - 
                                 data['volume_acceleration'].rolling(window=5).mean())
    
    # Regime Co-break Signal
    data['regime_co_break'] = data['volatility_regime_break'] * data['volume_regime_break']
    
    # Microstructure Pressure
    # Price Impact Resistance
    data['price_impact_resistance'] = (abs(data['close'] - data['open']) / 
                                     (data['high'] - data['low'] + 1e-6))
    
    # Volume-Weighted Pressure
    data['volume_weighted_pressure'] = data['amount'] / (data['high'] - data['low'] + 1e-6)
    
    # Microstructure Efficiency
    data['microstructure_efficiency'] = (data['price_impact_resistance'] / 
                                       (data['volume_weighted_pressure'] + 1e-6))
    
    # Alpha Integration
    # Regime Interaction Core
    data['regime_interaction_core'] = data['volume_volatility_coupling'] * data['regime_co_break']
    
    # Gap Efficiency Enhancement
    data['gap_efficiency_enhancement'] = data['regime_interaction_core'] * data['gap_asymmetry']
    
    # Final Alpha Factor
    alpha_factor = data['gap_efficiency_enhancement'] * data['microstructure_efficiency']
    
    # Return the alpha factor series with proper indexing
    return alpha_factor
