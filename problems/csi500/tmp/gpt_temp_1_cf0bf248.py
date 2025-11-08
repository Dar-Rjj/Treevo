import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Momentum Strength
    # Normalized Intraday Move
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['daily_range'] = (data['high'] - data['low']) / data['open']
    data['normalized_move'] = np.where(
        data['daily_range'] != 0,
        (data['close'] - data['open']) / (data['high'] - data['low']),
        0
    )
    
    # Momentum Persistence
    # 3-day Momentum Consistency
    data['intraday_move_sign'] = np.sign(data['close'] - data['open'])
    data['sign_consistency_3d'] = data['intraday_move_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: len(set(x)) == 1 if len(x) == 3 else 0
    ).fillna(0)
    
    # Magnitude persistence (3-day coefficient of variation)
    data['intraday_move_magnitude'] = abs(data['intraday_return'])
    data['magnitude_persistence'] = data['intraday_move_magnitude'].rolling(window=3, min_periods=1).std() / \
                                   (data['intraday_move_magnitude'].rolling(window=3, min_periods=1).mean() + 1e-8)
    data['magnitude_persistence'] = 1 / (1 + data['magnitude_persistence'])
    
    # Recent Momentum Acceleration
    data['momentum_1d'] = data['intraday_return']
    data['momentum_3d'] = data['intraday_return'].rolling(window=3, min_periods=1).mean()
    data['momentum_acceleration'] = np.where(
        data['momentum_1d'] * data['momentum_3d'] > 0,
        abs(data['momentum_1d']) / (abs(data['momentum_3d']) + 1e-8),
        0
    )
    
    # 2. Volatility Regime Detection
    # Short-term Volatility Measurement
    data['daily_range_vol'] = (data['high'] - data['low']) / data['open']
    data['volatility_5d'] = data['daily_range_vol'].rolling(window=5, min_periods=1).std()
    data['volatility_20d'] = data['daily_range_vol'].rolling(window=20, min_periods=1).std()
    
    # Volatility Regime Classification
    data['vol_percentile_20d'] = data['volatility_5d'].rolling(window=20, min_periods=1).apply(
        lambda x: (x[-1] > np.percentile(x, 60)) if len(x) == 20 else 0
    ).fillna(0)
    
    data['vol_regime'] = 0  # Neutral
    data.loc[data['vol_percentile_20d'] > 0.6, 'vol_regime'] = 1  # High volatility
    data.loc[data['vol_percentile_20d'] < 0.4, 'vol_regime'] = -1  # Low volatility
    
    # 3. Volume Confirmation Framework
    # Volume Trend Analysis
    data['volume_ma5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_ma5'] + 1e-8)
    
    # Volume-Price Alignment
    data['volume_price_alignment'] = np.where(
        data['intraday_return'] * data['volume_ratio'] > 0,
        abs(data['volume_ratio']),
        -abs(data['volume_ratio'])
    )
    
    # Relative Volume Strength
    data['volume_percentile_20d'] = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) if len(x) == 20 else 0.5
    ).fillna(0.5)
    
    # Volume Quality Assessment
    data['volume_trend_3d'] = data['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: 1 if len(x) == 3 and x.iloc[0] < x.iloc[1] < x.iloc[2] else 0
    ).fillna(0)
    
    # 4. Adaptive Signal Combination
    # Base momentum signal
    base_momentum = data['normalized_move']
    
    # Volatility regime adjustments
    volatility_score = data['volatility_5d'] / (data['volatility_20d'] + 1e-8)
    confidence_score = 1 / (1 + data['volatility_5d'])
    
    regime_adjusted_momentum = base_momentum.copy()
    high_vol_mask = data['vol_regime'] == 1
    low_vol_mask = data['vol_regime'] == -1
    
    regime_adjusted_momentum[high_vol_mask] = base_momentum[high_vol_mask] * (1 - volatility_score[high_vol_mask])
    regime_adjusted_momentum[low_vol_mask] = base_momentum[low_vol_mask] * (1 + confidence_score[low_vol_mask])
    
    # Volume confirmation multiplier
    volume_multiplier = 1 + data['volume_ratio'] * data['volume_price_alignment'] * 0.5
    volume_multiplier = np.clip(volume_multiplier, 0.5, 2.0)
    
    # Momentum persistence enhancement
    persistence_enhancement = 1 + (data['sign_consistency_3d'] * 0.3 + 
                                 data['magnitude_persistence'] * 0.2 + 
                                 data['momentum_acceleration'] * 0.2)
    
    # Final factor construction
    final_factor = (regime_adjusted_momentum * 
                   volume_multiplier * 
                   persistence_enhancement * 
                   (1 + data['volume_percentile_20d'] * 0.5))
    
    return final_factor
