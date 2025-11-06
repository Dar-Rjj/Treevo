import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Gap Momentum Dynamics
    # Overnight Gap Analysis
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_decay'] = data['overnight_gap'] - data['overnight_gap'].rolling(window=3, min_periods=1).sum()
    
    # Intraday Gap Behavior
    gap_denominator = np.abs(data['open'] - data['close'].shift(1))
    gap_denominator = gap_denominator.replace(0, np.nan)
    data['gap_fill_momentum'] = (data['close'] - data['open']) / gap_denominator
    data['gap_persistence'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Multi-Timeframe Integration
    gap_direction = np.sign(data['open'] - data['close'].shift(1))
    data['gap_direction_consistency'] = gap_direction.rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    data['gap_decay_intensity'] = data['gap_decay'] * data['gap_persistence']
    
    # Efficiency-Weighted Volume Anchoring
    # Price Efficiency Metrics
    data['range_5d'] = data['high'].rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x - data.loc[x.index, 'low']) if len(x) > 0 else np.nan, raw=False
    )
    data['efficiency_5d'] = np.abs(data['close'] - data['close'].shift(5)) / data['range_5d']
    
    data['range_10d'] = data['high'].rolling(window=10, min_periods=1).apply(
        lambda x: np.sum(x - data.loc[x.index, 'low']) if len(x) > 0 else np.nan, raw=False
    )
    data['efficiency_10d'] = np.abs(data['close'] - data['close'].shift(10)) / data['range_10d']
    data['efficiency_momentum'] = data['efficiency_5d'] - data['efficiency_10d']
    
    # Volume Pressure Components
    data['volume_sma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_deviation'] = data['volume'] - data['volume_sma_5']
    
    price_range = data['high'] - data['low']
    price_range = price_range.replace(0, np.nan)
    data['volume_concentration'] = data['amount'] / price_range
    
    # Efficiency-Weighted Enhancement
    data['volume_anchoring'] = data['volume_deviation'] * (data['close'] - data['low']) / price_range
    data['efficiency_pressure'] = (data['close'] - data['open']) / data['volume'] * data['efficiency_momentum']
    
    # Volatility-Regime Adaptation
    # Volatility Context
    data['high_5d'] = data['high'].rolling(window=5, min_periods=1).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=1).min()
    data['range_ratio'] = (data['high'] - data['low']) / (data['high_5d'] - data['low_5d'])
    
    data['volatility_5d'] = data['close'].rolling(window=5, min_periods=1).std()
    data['volatility_20d'] = data['close'].rolling(window=20, min_periods=1).std()
    volatility_denominator = data['volatility_20d'].replace(0, np.nan)
    data['volatility_ratio'] = data['volatility_5d'] / volatility_denominator
    
    # Regime-Specific Processing
    data['regime_multiplier'] = np.where(data['range_ratio'] > 1.1, 1.3, 
                                       np.where(data['range_ratio'] < 0.9, 0.6, 1.0))
    data['volume_confirmation'] = data['volume_anchoring'] * data['gap_direction_consistency']
    
    # Composite Alpha Construction
    # Core Component Integration
    data['regime_adapted_gap_momentum'] = data['gap_decay_intensity'] * data['regime_multiplier']
    data['efficiency_weighted_volume'] = data['volume_anchoring'] * data['efficiency_momentum']
    
    # Multi-Factor Synthesis
    data['momentum_base'] = data['regime_adapted_gap_momentum'] * data['efficiency_weighted_volume']
    data['volume_confirmation_final'] = data['momentum_base'] * data['volume_concentration'] * data['efficiency_momentum']
    
    # Final Alpha Output
    data['composite_alpha'] = data['momentum_base'] * data['volume_confirmation_final']
    data['final_alpha'] = data['composite_alpha'].rolling(window=3, min_periods=1).mean()
    
    return data['final_alpha']
