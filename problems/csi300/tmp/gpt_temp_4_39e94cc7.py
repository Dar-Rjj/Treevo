import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Asymmetric Volatility Momentum
    # Multi-timeframe Momentum
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    data['mom_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Directional Divergence
    data['price_dir'] = np.sign(data['close'] - data['close'].shift(3))
    data['volume_dir'] = np.sign(data['volume'] - data['volume'].shift(3))
    data['divergence'] = data['price_dir'] * data['volume_dir']
    
    # Volatility Asymmetry
    # Calculate returns for volatility
    data['high_ret'] = data['high'] / data['close'].shift(1) - 1
    data['low_ret'] = data['low'] / data['close'].shift(1) - 1
    
    # Rolling upside and downside volatility
    upside_vol = data['high_ret'].rolling(window=10, min_periods=5).apply(
        lambda x: np.std(x[x > 0]) if len(x[x > 0]) > 2 else 1.0
    )
    downside_vol = data['low_ret'].rolling(window=10, min_periods=5).apply(
        lambda x: np.std(x[x < 0]) if len(x[x < 0]) > 2 else 1.0
    )
    
    # Combined asymmetric volatility momentum
    momentum_alignment = (data['mom_3d'] + data['mom_10d'] + data['mom_20d']) / 3
    vol_ratio = upside_vol / downside_vol
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    asym_vol_momentum = momentum_alignment * vol_ratio * data['divergence']
    
    # Volume-Weighted Range Breakout
    # Breakout Conditions
    data['daily_range'] = data['high'] - data['low']
    range_ma_20 = data['daily_range'].rolling(window=20, min_periods=10).mean()
    data['range_ratio'] = data['daily_range'] / range_ma_20
    
    volume_ma_20 = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_cluster'] = (data['volume'] > 1.5 * volume_ma_20).astype(float)
    
    # Intraday Pressure
    volume_ma_3 = data['volume'].rolling(window=3, min_periods=2).mean()
    data['opening_pressure'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * \
                              (data['volume'] / volume_ma_3)
    data['closing_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) * \
                              (data['volume'] / volume_ma_3)
    
    # Breakout Strength
    high_rolling_5 = data['high'].rolling(window=5, min_periods=3).max()
    data['breakout_strength'] = (data['close'] > high_rolling_5.shift(1)).astype(float)
    
    # Weighted breakout signal
    breakout_signal = data['breakout_strength'] * data['range_ratio'] * \
                     data['volume_cluster'] * (data['opening_pressure'] + data['closing_pressure'])
    
    # Intraday Efficiency Momentum
    # Efficiency Patterns
    data['morning_strength'] = data['high'] / data['open'] - 1
    data['afternoon_efficiency'] = data['close'] / data['high'] - 1
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['daily_efficiency'] = data['daily_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Elasticity
    data['price_bounce'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_bounce'] = data['price_bounce'].replace([np.inf, -np.inf], 0).fillna(0)
    data['volume_elasticity'] = data['volume'] / abs(data['close'] - data['open'])
    data['volume_elasticity'] = data['volume_elasticity'].replace([np.inf, -np.inf], 0).fillna(0)
    data['elasticity_factor'] = data['price_bounce'] * data['volume_elasticity']
    
    # Efficiency Momentum
    efficiency_ma_5 = data['daily_efficiency'].rolling(window=5, min_periods=3).mean()
    data['efficiency_momentum'] = data['daily_efficiency'] / efficiency_ma_5.shift(1)
    data['efficiency_momentum'] = data['efficiency_momentum'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Combined efficiency signal
    elasticity_acceleration = data['elasticity_factor'] / data['elasticity_factor'].shift(1)
    elasticity_acceleration = elasticity_acceleration.replace([np.inf, -np.inf], 1).fillna(1)
    efficiency_signal = data['efficiency_momentum'] * elasticity_acceleration
    
    # Liquidity-Adjusted Gap Reversal
    # Gap Signal
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_magnitude'] = abs(data['opening_gap'])
    
    # Range Expansion
    data['current_range'] = (data['high'] - data['low']) / data['close']
    range_ma_5 = data['current_range'].rolling(window=5, min_periods=3).mean()
    data['range_momentum'] = data['current_range'] / range_ma_5.shift(1)
    data['range_expansion'] = data['range_momentum'] * data['volume']
    
    # Reversal Signal
    data['liquidity_measure'] = data['amount'] / data['volume']
    data['liquidity_measure'] = data['liquidity_measure'].replace([np.inf, -np.inf], 1).fillna(1)
    reversal_signal = -data['gap_magnitude'] * data['liquidity_measure'] * data['range_expansion']
    
    # Regime-Adaptive Order Flow
    # Volatility Regime
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_ma_10 = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['regime_shift'] = (data['true_range'] > 1.5 * tr_ma_10.shift(1)).astype(float)
    
    # Acceleration Divergence
    data['price_acceleration'] = (data['close'] - data['close'].shift(1)) - \
                                (data['close'].shift(1) - data['close'].shift(2))
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(1)) - \
                                 (data['volume'].shift(1) - data['volume'].shift(2))
    data['acceleration_divergence'] = data['price_acceleration'] * data['volume_acceleration']
    
    # Adaptive Signal
    signed_volume_window = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: np.sum(x * np.sign(data.loc[x.index, 'close'] - data.loc[x.index, 'open']))
    )
    regime_signal = signed_volume_window * data['regime_shift'] * data['acceleration_divergence']
    
    # Combine all factors with equal weights
    result = (asym_vol_momentum + breakout_signal + efficiency_signal + 
             reversal_signal + regime_signal) / 5
    
    # Clean any remaining infinite or NaN values
    result = result.replace([np.inf, -np.inf], 0).fillna(0)
    
    return result
