import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Volatility Clustering Divergence
    # Calculate ATR (Average True Range)
    data['tr'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    data['atr_5'] = data['tr'].rolling(window=5).mean()
    data['atr_10'] = data['tr'].rolling(window=10).mean()
    data['atr_20'] = data['tr'].rolling(window=20).mean()
    data['atr_50'] = data['tr'].rolling(window=50).mean()
    
    volatility_divergence = (data['atr_5'] / data['atr_10']) - (data['atr_20'] / data['atr_50'])
    
    # 2. Price-Volume Efficiency Ratio
    data['price_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_efficiency'] = data['volume'] / data['volume_5d_avg']
    price_volume_efficiency = data['price_efficiency'] * data['volume_efficiency']
    
    # 3. Extreme Reversal Indicator
    data['prev_close'] = data['close'].shift(1)
    data['high_10d'] = data['high'].rolling(window=10).max()
    data['low_10d'] = data['low'].rolling(window=10).min()
    data['volume_10d_avg'] = data['volume'].rolling(window=10).mean()
    
    is_extreme_high = data['close'] == data['high_10d']
    is_extreme_low = data['close'] == data['low_10d']
    volume_confirmation = data['volume'] > (2 * data['volume_10d_avg'])
    
    extreme_reversal = -np.sign(data['close'] - data['prev_close']) * (data['volume'] / data['volume_10d_avg'])
    extreme_reversal = extreme_reversal.where((is_extreme_high | is_extreme_low) & volume_confirmation, 0)
    
    # 4. Momentum Persistence Score
    data['return_direction'] = np.sign(data['returns'])
    data['persistence_days'] = 0
    
    for i in range(1, len(data)):
        if data['return_direction'].iloc[i] == data['return_direction'].iloc[i-1]:
            data['persistence_days'].iloc[i] = data['persistence_days'].iloc[i-1] + 1
    
    data['persistence_volume_avg'] = data['volume'].rolling(window=5).mean()
    momentum_persistence = data['persistence_days'] * (data['volume'] / data['persistence_volume_avg'])
    
    # 5. Gap Fill Probability
    data['gap_size'] = abs(data['open'] - data['prev_close'])
    data['intraday_range'] = data['high'] - data['low']
    gap_fill = -(data['gap_size'] / data['intraday_range'].replace(0, np.nan)) * data['volume']
    
    # 6. Volume Accumulation Divergence
    data['up_day'] = data['close'] > data['open']
    data['down_day'] = data['close'] < data['open']
    
    up_volume_days = data['up_day'].rolling(window=10).sum()
    down_volume_days = data['down_day'].rolling(window=10).sum()
    
    volume_accumulation = (up_volume_days - down_volume_days) * (data['volume'] / data['volume_10d_avg'])
    
    # 7. Price Compression Breakout
    data['daily_range'] = data['high'] - data['low']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5).mean()
    data['range_20d_avg'] = data['daily_range'].rolling(window=20).mean()
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    
    range_compression = data['range_5d_avg'] / data['range_20d_avg']
    volume_spike = data['volume'] > (2 * data['volume_20d_avg'])
    
    price_compression = range_compression * volume_spike.astype(int) * np.sign(data['close'] - data['open'])
    
    # 8. Multi-Timeframe Momentum Alignment
    data['momentum_3d'] = data['close'] / data['close'].shift(3)
    data['momentum_10d'] = data['close'] / data['close'].shift(10)
    data['momentum_20d'] = data['close'] / data['close'].shift(20)
    
    momentum_alignment = (data['momentum_3d'] + data['momentum_10d'] + data['momentum_20d']) / 3
    
    # 9. Volume-Weighted Price Stability
    data['returns_5d_std'] = data['returns'].rolling(window=5).std()
    price_stability = 1 / data['returns_5d_std'].replace(0, np.nan)
    volume_trend = data['volume'] / data['volume_10d_avg']
    
    volume_weighted_stability = price_stability * volume_trend
    
    # 10. Opening Gap Momentum
    gap_direction = np.sign(data['open'] - data['prev_close'])
    gap_strength = abs(data['open'] - data['prev_close']) / data['prev_close']
    intraday_momentum = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    gap_momentum = gap_direction * gap_strength * intraday_momentum
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'volatility_divergence': volatility_divergence,
        'price_volume_efficiency': price_volume_efficiency,
        'extreme_reversal': extreme_reversal,
        'momentum_persistence': momentum_persistence,
        'gap_fill': gap_fill,
        'volume_accumulation': volume_accumulation,
        'price_compression': price_compression,
        'momentum_alignment': momentum_alignment,
        'volume_stability': volume_weighted_stability,
        'gap_momentum': gap_momentum
    })
    
    # Z-score normalize each factor and take average
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std())
    final_factor = normalized_factors.mean(axis=1)
    
    return final_factor
