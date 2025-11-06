import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Asymmetry Components
    data['up_move_asymmetry'] = np.where(
        data['open'] != data['low'],
        ((data['high'] - data['open']) / (data['open'] - data['low'])) * data['volume'],
        0
    )
    data['down_move_asymmetry'] = np.where(
        data['high'] != data['open'],
        ((data['open'] - data['low']) / (data['high'] - data['open'])) * data['volume'],
        0
    )
    data['close_bias'] = np.where(
        data['high'] != data['low'],
        ((data['close'] - data['open']) / (data['high'] - data['low'])) * data['amount'],
        0
    )
    
    # Volatility Structure
    data['close_ret'] = data['close'] / data['close'].shift(1) - 1
    data['realized_vol'] = data['close_ret'].rolling(window=5).apply(
        lambda x: np.sqrt(np.nansum(x**2)), raw=False
    )
    data['expected_vol'] = (data['high'] - data['low']) / data['close']
    data['volatility_surprise'] = data['realized_vol'] / data['expected_vol']
    
    # Volume Dynamics
    data['volume_momentum'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    data['volume_acceleration'] = data['volume_momentum'] - data['volume_momentum'].shift(1)
    data['volume_persistence'] = (
        (data['volume'] > data['volume'].shift(1)).rolling(window=5).sum()
    )
    
    # Price-Volume Divergence
    data['short_divergence'] = (data['close'] / data['close'].shift(1) - 1) - (data['volume'] / data['volume'].shift(1) - 1)
    data['medium_divergence'] = (data['close'] / data['close'].shift(5) - 1) - (data['volume'] / data['volume'].shift(5) - 1)
    data['long_divergence'] = (data['close'] / data['close'].shift(20) - 1) - (data['volume'] / data['volume'].shift(20) - 1)
    
    # Microstructure Patterns
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['closing_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_reversal'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Asymmetry-Volatility Interaction
    data['vol_scaled_up_asymmetry'] = data['up_move_asymmetry'] * data['volatility_surprise']
    data['vol_scaled_down_asymmetry'] = data['down_move_asymmetry'] / data['volatility_surprise']
    data['vol_adjusted_close_bias'] = data['close_bias'] * (1 + np.abs(data['volatility_surprise'] - 1))
    
    # Volume-Regime Classification
    data['high_volume_regime'] = (data['volume'] > data['volume'].shift(1)) & (data['volume'] > data['volume'].shift(2))
    data['low_volume_regime'] = (data['volume'] < data['volume'].shift(1)) & (data['volume'] < data['volume'].shift(2))
    data['transition_volume'] = ~(data['high_volume_regime'] | data['low_volume_regime'])
    
    # Regime-Specific Factors
    data['high_volume_alpha'] = data['vol_scaled_up_asymmetry'] * data['volume_momentum']
    data['low_volume_alpha'] = data['vol_scaled_down_asymmetry'] * data['volume_persistence']
    data['transition_alpha'] = data['vol_adjusted_close_bias'] * data['volume_acceleration']
    
    # Divergence Enhancement
    data['enhanced_short'] = data['short_divergence'] * data['opening_pressure']
    data['enhanced_medium'] = data['medium_divergence'] * data['closing_pressure']
    data['enhanced_long'] = data['long_divergence'] * data['intraday_reversal']
    
    # Composite Alpha Generation
    data['regime_blended'] = (
        data['high_volume_alpha'] * data['high_volume_regime'] +
        data['low_volume_alpha'] * data['low_volume_regime'] +
        data['transition_alpha'] * data['transition_volume']
    )
    data['divergence_weighted'] = data['regime_blended'] * (data['enhanced_short'] + data['enhanced_medium'] + data['enhanced_long'])
    data['microstructure_refined'] = data['divergence_weighted'] * (1 + np.abs(data['intraday_reversal']))
    data['final_alpha'] = data['microstructure_refined'] * (data['volume'] / data['volume'].shift(1))
    
    return data['final_alpha']
