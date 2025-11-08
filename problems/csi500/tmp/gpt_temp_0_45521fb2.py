import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price-Volume Interactions
    # Volume-Accelerated Momentum Divergence
    data['price_momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['momentum_divergence'] = data['price_momentum_3d'] - data['volume_momentum_3d']
    
    # Range Expansion Efficiency
    data['current_range'] = (data['high'] - data['low']) / data['close']
    data['avg_range_5d'] = ((data['high'] - data['low']) / data['close']).rolling(window=5).mean().shift(1)
    data['range_expansion_ratio'] = data['current_range'] / data['avg_range_5d']
    
    # Gap Persistence Strength
    data['gap_direction'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['open'].shift(1) - data['close'].shift(2))
    data['volume_persistence'] = data['volume'] / data['volume'].shift(1)
    data['gap_strength'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = data['gap_direction'] * data['volume_persistence'] * data['gap_strength']
    
    # Microstructure Efficiency Patterns
    # Price Discovery Quality
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['efficiency_persistence'] = data['intraday_efficiency'].rolling(window=5).apply(
        lambda x: np.corrcoef(x.values, range(5))[0,1] if len(x) == 5 and not x.isna().any() else np.nan
    )
    data['efficiency_quality'] = data['intraday_efficiency'] * data['efficiency_persistence']
    
    # Volume-Weighted Price Deviation
    data['vwap'] = data['amount'] / data['volume']
    data['price_deviation'] = (data['close'] - data['vwap']) / data['close']
    data['volume_confirmation'] = data['volume'] / data['volume'].rolling(window=5).mean().shift(1)
    data['weighted_deviation'] = data['price_deviation'] * data['volume_confirmation']
    
    # Opening Session Momentum Flow
    data['opening_gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_momentum_persistence'] = (data['close'] - data['open']) / (data['close'].shift(1) - data['open'].shift(1))
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(1)
    data['opening_flow'] = data['opening_gap_momentum'] * data['intraday_momentum_persistence'] * data['volume_acceleration']
    
    # Acceleration and Reversal Dynamics
    # Momentum Volume Convergence
    data['price_acceleration'] = (data['close'] / data['close'].shift(5)) / (data['close'].shift(5) / data['close'].shift(10))
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(5)) / (data['volume'].shift(5) / data['volume'].shift(10))
    data['acceleration_convergence'] = data['price_acceleration'] - data['volume_acceleration']
    
    # Volume Breakout Confirmation
    data['price_level'] = data['close'] / data['high'].rolling(window=10).max().shift(1)
    data['volume_breakout'] = data['volume'] / data['volume'].rolling(window=10).max().shift(1)
    data['breakout_strength'] = data['price_level'] * data['volume_breakout']
    
    # Range Compression Expansion Cycle
    data['range_compression'] = data['current_range'] / data['current_range'].rolling(window=5).mean().shift(1)
    data['volume_compression'] = data['volume'] / data['volume'].rolling(window=5).mean().shift(1)
    data['compression_divergence'] = data['range_compression'] - data['volume_compression']
    
    # Persistence and Flow Dynamics
    # Volume-Confirmed Mean Reversion
    data['price_deviation_trend'] = data['close'] / data['close'].rolling(window=5).mean().shift(1)
    data['volume_trend_confirmation'] = data['volume'] / data['volume'].rolling(window=5).mean().shift(1)
    data['reversion_signal'] = data['price_deviation_trend'] / data['volume_trend_confirmation']
    
    # Accumulation Distribution Efficiency
    data['money_flow'] = ((data['high'] + data['low'] + data['close']) / 3) * data['volume']
    data['flow_persistence'] = data['money_flow'].rolling(window=5).std() / data['money_flow'].rolling(window=5).mean()
    data['price_flow_efficiency'] = (data['close'] / data['close'].shift(5)) / (data['money_flow'].rolling(window=5).sum() / data['money_flow'].shift(5))
    data['flow_efficiency_score'] = data['flow_persistence'] * data['price_flow_efficiency']
    
    # Breakout Sustainability
    data['price_breakout'] = data['close'] / data['high'].rolling(window=10).max().shift(1)
    data['volume_sustainability'] = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    data['range_expansion_confirmation'] = (data['high'] - data['low']) / ((data['high'] - data['low']).rolling(window=5).mean().shift(1))
    data['sustainable_breakout'] = data['price_breakout'] * data['volume_sustainability'] * data['range_expansion_confirmation']
    
    # Combine all factors using equal weighting
    factors = [
        'momentum_divergence', 'range_expansion_ratio', 'gap_persistence',
        'efficiency_quality', 'weighted_deviation', 'opening_flow',
        'acceleration_convergence', 'breakout_strength', 'compression_divergence',
        'reversion_signal', 'flow_efficiency_score', 'sustainable_breakout'
    ]
    
    # Create final factor by averaging all normalized components
    final_factor = pd.Series(index=data.index, dtype=float)
    for factor in factors:
        if factor in data.columns:
            # Z-score normalization
            normalized = (data[factor] - data[factor].rolling(window=20).mean()) / data[factor].rolling(window=20).std()
            final_factor = final_factor.add(normalized, fill_value=0)
    
    # Average all components
    final_factor = final_factor / len(factors)
    
    return final_factor
