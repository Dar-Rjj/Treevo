import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Divergence Detection
    # 5-day vs 20-day price momentum difference
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_divergence'] = data['momentum_5d'] - data['momentum_20d']
    
    # Volume Asymmetry Analysis
    # Compute up-volume/down-volume ratio
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['up_volume'] = np.where(data['price_change'] > 0, data['volume'], 0)
    data['down_volume'] = np.where(data['price_change'] < 0, data['volume'], 0)
    
    # Rolling 10-day up/down volume sums
    data['up_volume_sum'] = data['up_volume'].rolling(window=10, min_periods=5).sum()
    data['down_volume_sum'] = data['down_volume'].rolling(window=10, min_periods=5).sum()
    data['volume_asymmetry'] = (data['up_volume_sum'] - data['down_volume_sum']) / (data['up_volume_sum'] + data['down_volume_sum'] + 1e-8)
    
    # Volume sensitivity to price movements
    data['volume_sensitivity'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean() * np.abs(data['price_change'] / data['close'].shift(1))
    
    # Range Efficiency Scoring
    # Daily price range percentage
    data['daily_range_pct'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # 20-day volatility (using close-to-close returns)
    data['returns'] = data['close'].pct_change()
    data['volatility_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    # Range efficiency: daily range relative to volatility
    data['range_efficiency'] = data['daily_range_pct'] / (data['volatility_20d'] + 1e-8)
    
    # Breakout detection using range efficiency
    data['range_efficiency_ma'] = data['range_efficiency'].rolling(window=10, min_periods=5).mean()
    data['range_breakout'] = data['range_efficiency'] - data['range_efficiency_ma']
    
    # Composite Signal Generation
    # Weight momentum divergence by volume asymmetry
    momentum_volume_weighted = data['momentum_divergence'] * (1 + data['volume_asymmetry'])
    
    # Enhance with range efficiency breakout signals
    composite_signal = momentum_volume_weighted * (1 + data['range_breakout']) * (1 + data['volume_sensitivity'])
    
    # Final factor with normalization
    factor = composite_signal.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 0
    )
    
    return factor
