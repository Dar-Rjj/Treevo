import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum Reversal with Volume-Efficiency Confirmation
    """
    data = df.copy()
    
    # Volatility Regime Classification
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Compute 10-day Average True Range
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=10).mean()
    
    # Compute 60-day Median ATR
    data['atr_60_median'] = data['true_range'].rolling(window=60, min_periods=60).median()
    
    # Volatility regime classification
    data['volatility_weight'] = data['atr_10'] / data['atr_60_median']
    
    # Momentum Reversal Detection
    # Compute Short-Term Momentum Divergence
    data['sma_5'] = data['close'].rolling(window=5, min_periods=5).mean()
    data['sma_20'] = data['close'].rolling(window=20, min_periods=20).mean()
    data['momentum_ratio'] = data['sma_5'] / data['sma_20']
    
    # Detect Intraday Reversal Efficiency
    data['actual_range'] = data['high'] - data['low']
    data['range_efficiency'] = (data['close'] - data['open']) / data['actual_range']
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate Reversal Magnitude
    data['price_change'] = abs(data['close'] - data['open'])
    data['reversal_direction'] = np.sign(data['close'] - data['open'])
    data['reversal_magnitude'] = data['price_change'] * data['reversal_direction']
    
    # Volume Confirmation with Persistence
    # Calculate Volume Momentum
    data['volume_sma_5'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['volume_sma_20'] = data['volume'].rolling(window=20, min_periods=20).mean()
    data['volume_ratio'] = data['volume_sma_5'] / data['volume_sma_20']
    
    # Calculate Volume Persistence
    volume_condition = data['volume_ratio'] > 1.05
    data['volume_persistence'] = volume_condition.astype(int)
    data['volume_persistence_count'] = data['volume_persistence'].rolling(window=20, min_periods=1).apply(
        lambda x: (x == 1).cumsum().iloc[-1] if len(x) > 0 else 0, raw=False
    )
    
    # Volatility-Regime Specific Enhancement
    # High Volatility Strategy
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['return_1d'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_acceleration'] = (data['return_5d'] - data['return_1d']) * data['volume_ratio']
    
    # Range Breakout Signal for High Volatility
    data['high_5d_max'] = data['high'].shift(1).rolling(window=5, min_periods=5).max()
    data['range_breakout'] = data['close'] / data['high_5d_max'] - 1
    
    # Low Volatility Strategy
    data['close_20d_avg'] = data['close'].shift(1).rolling(window=20, min_periods=20).mean()
    data['mean_reversion_potential'] = data['close'] / data['close_20d_avg'] - 1
    
    # Volume Breakout for Low Volatility
    data['volume_20d_max'] = data['volume'].shift(1).rolling(window=20, min_periods=20).max()
    data['volume_breakout'] = data['volume'] / data['volume_20d_max'] - 1
    
    # High Volatility Strategy Blend
    data['hv_strategy'] = (data['momentum_acceleration'] + data['range_breakout']) / 2
    
    # Low Volatility Strategy Blend
    data['lv_strategy'] = (data['mean_reversion_potential'] + data['volume_breakout']) / 2
    
    # Final Alpha Construction
    # Base conditions
    momentum_condition = data['momentum_ratio'] < 0.95
    volume_condition = data['volume_ratio'] > 1.05
    persistence_condition = data['volume_persistence_count'] >= 2
    
    # Base factor calculation
    base_factor = (1 - data['momentum_ratio']) * data['reversal_magnitude'] * data['range_efficiency'] * data['volume_ratio'] * data['volume_persistence_count']
    
    # Volatility-adaptive enhancement
    volatility_enhancement = data['hv_strategy'] * data['volatility_weight'] + data['lv_strategy'] * (1 - data['volatility_weight'])
    
    # Final alpha with conditions
    alpha = pd.Series(0, index=data.index)
    valid_condition = momentum_condition & volume_condition & persistence_condition
    
    alpha[valid_condition] = (
        base_factor[valid_condition] * 
        volatility_enhancement[valid_condition] * 
        data['reversal_direction'][valid_condition]
    )
    
    return alpha
