import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Decay Asymmetry Factor
    Combines momentum persistence analysis with decay asymmetry detection
    to predict future returns based on asymmetric momentum decay patterns.
    """
    data = df.copy()
    
    # Raw Momentum Components
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['range_momentum'] = ((data['high'] - data['low']) / 
                             (data['high'].shift(1) - data['low'].shift(1)) - 1)
    
    # Momentum Acceleration and Decay
    data['price_acceleration'] = data['price_momentum'] / data['price_momentum'].shift(1) - 1
    data['volume_acceleration'] = data['volume_momentum'] / data['volume_momentum'].shift(1) - 1
    
    # Asymmetry Detection Components
    # Up vs Down Momentum Characteristics
    data['up_momentum'] = np.where(data['price_momentum'] > 0, data['price_momentum'], 0)
    data['down_momentum'] = np.where(data['price_momentum'] < 0, abs(data['price_momentum']), 0)
    
    # Opening vs Closing Asymmetry
    data['morning_momentum'] = (data['high'] - data['open']) / data['open']
    data['afternoon_momentum'] = (data['close'] - data['low']) / data['close']
    
    # Range Utilization Asymmetry
    data['up_range_util'] = np.where(data['close'] > data['open'], 
                                   (data['close'] - data['open']) / (data['high'] - data['low']), 0)
    data['down_range_util'] = np.where(data['close'] < data['open'], 
                                     (data['open'] - data['close']) / (data['high'] - data['low']), 0)
    
    # Decay Rate Measurement
    # Multi-timeframe decay analysis
    for window in [3, 5]:
        data[f'price_momentum_ma_{window}'] = data['price_momentum'].rolling(window).mean()
        data[f'volume_momentum_ma_{window}'] = data['volume_momentum'].rolling(window).mean()
    
    # Volume-Decay Relationship
    data['volume_price_decay_ratio'] = (data['volume_acceleration'].rolling(3).std() / 
                                      (data['price_acceleration'].rolling(3).std() + 1e-8))
    
    # Range Impact on Decay
    data['daily_range'] = data['high'] - data['low']
    data['range_expansion'] = data['daily_range'] / data['daily_range'].shift(1) - 1
    
    # Asymmetric Signal Generation
    # Momentum persistence scoring
    data['up_momentum_persistence'] = (data['up_momentum'].rolling(5).sum() / 
                                     (abs(data['price_momentum']).rolling(5).sum() + 1e-8))
    data['down_momentum_persistence'] = (data['down_momentum'].rolling(5).sum() / 
                                       (abs(data['price_momentum']).rolling(5).sum() + 1e-8))
    
    # Decay-Volume Alignment
    data['decay_volume_alignment'] = (data['price_acceleration'].rolling(3).corr(data['volume_acceleration']) * 
                                    np.sign(data['price_momentum']))
    
    # Market Condition Adaptation
    # Volatility-based adjustments
    volatility_20d = data['close'].pct_change().rolling(20).std()
    data['volatility_adjusted_decay'] = data['price_acceleration'] / (volatility_20d + 1e-8)
    
    # Trend environment classification
    trend_strength = data['close'].rolling(10).apply(lambda x: (x[-1] - x[0]) / (x.std() + 1e-8))
    data['trend_adjusted_momentum'] = data['price_momentum'] * np.sign(trend_strength)
    
    # Factor Integration
    # Asymmetry Score Calculation
    data['momentum_decay_asymmetry'] = (
        (data['up_momentum_persistence'] - data['down_momentum_persistence']) *
        data['volatility_adjusted_decay'] *
        (data['up_range_util'] - data['down_range_util'])
    )
    
    # Multi-timeframe persistence integration
    data['multi_timeframe_persistence'] = (
        data['price_momentum_ma_3'] * 0.4 +
        data['price_momentum_ma_5'] * 0.6
    )
    
    # Final Alpha Generation
    data['momentum_decay_factor'] = (
        data['momentum_decay_asymmetry'] * 0.5 +
        data['multi_timeframe_persistence'] * 0.3 +
        data['decay_volume_alignment'] * 0.2
    )
    
    # Signal strength assessment and smoothing
    factor = (data['momentum_decay_factor'].rolling(3).mean() * 
             np.sign(data['trend_adjusted_momentum']))
    
    return factor
