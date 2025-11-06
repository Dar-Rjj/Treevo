import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for volatility calculations
    data['returns'] = data['close'].pct_change()
    
    # Multi-Timeframe Acceleration Calculation
    # Short-term Acceleration (3-6 days)
    data['price_accel_short'] = (data['close'] / data['close'].shift(6) - 1) - (data['close'] / data['close'].shift(3) - 1)
    data['volume_accel_short'] = (data['volume'] / data['volume'].shift(6) - 1) - (data['volume'] / data['volume'].shift(3) - 1)
    
    # Medium-term Acceleration (6-12 days)
    data['price_accel_medium'] = (data['close'] / data['close'].shift(12) - 1) - (data['close'] / data['close'].shift(6) - 1)
    data['volume_accel_medium'] = (data['volume'] / data['volume'].shift(12) - 1) - (data['volume'] / data['volume'].shift(6) - 1)
    
    # Long-term Acceleration (10-20 days)
    data['price_accel_long'] = (data['close'] / data['close'].shift(20) - 1) - (data['close'] / data['close'].shift(10) - 1)
    data['volume_accel_long'] = (data['volume'] / data['volume'].shift(20) - 1) - (data['volume'] / data['volume'].shift(10) - 1)
    
    # Acceleration Divergence Analysis
    data['short_medium_divergence'] = data['price_accel_short'] - data['price_accel_medium']
    data['medium_long_divergence'] = data['price_accel_medium'] - data['price_accel_long']
    data['volume_divergence_signal'] = (data['volume_accel_short'] < 0) & (data['volume_accel_medium'] > 0)
    
    # Volatility Context Integration
    # True Range Calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['avg_true_range'] = data['true_range'].rolling(window=20).mean()
    data['volatility_weight'] = data['true_range'] / data['avg_true_range']
    
    # Volatility Regime Assessment
    data['recent_volatility'] = data['returns'].rolling(window=5).std()
    data['historical_volatility'] = data['returns'].rolling(window=20).std()
    data['volatility_classification'] = data['recent_volatility'] > data['historical_volatility']
    data['volatility_regime_change'] = data['volatility_classification'] != data['volatility_classification'].shift(1)
    
    # Volume Pattern Context
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(5) - 1) - (data['volume'].shift(5) / data['volume'].shift(10) - 1)
    
    # Volume Regime Detection
    data['volume_percentile_20'] = data['volume'].rolling(window=20).apply(lambda x: np.percentile(x, 70))
    data['volume_percentile_30'] = data['volume'].rolling(window=20).apply(lambda x: np.percentile(x, 30))
    data['high_volume_days'] = data['volume'] > data['volume_percentile_20']
    data['low_volume_days'] = data['volume'] < data['volume_percentile_30']
    
    # Multi-Timeframe Convergence
    data['momentum_short'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_alignment'] = np.sign(data['momentum_short']) == np.sign(data['momentum_medium'])
    data['volume_price_divergence'] = ((data['momentum_short'] > 0) & (data['volume_momentum'] < 0)) | ((data['momentum_short'] < 0) & (data['volume_momentum'] > 0))
    
    # Adaptive Alpha Generation
    # High Volatility Strategy
    high_vol_primary = data['short_medium_divergence'] > 0
    high_vol_confirmation = data['volume_divergence_signal'] & data['momentum_alignment']
    high_vol_strength = abs(data['price_accel_short']) * abs(data['price_accel_medium'])
    high_vol_factor = high_vol_primary.astype(int) * high_vol_confirmation.astype(int) * high_vol_strength * data['volatility_weight']
    
    # Low Volatility Strategy
    low_vol_primary = data['medium_long_divergence'] > 0
    low_vol_confirmation = data['volume_price_divergence'] & data['momentum_alignment']
    low_vol_strength = abs(data['price_accel_medium']) * abs(data['price_accel_long'])
    low_vol_factor = low_vol_primary.astype(int) * low_vol_confirmation.astype(int) * low_vol_strength
    
    # Combine strategies based on volatility regime
    final_factor = np.where(data['volatility_classification'], high_vol_factor, low_vol_factor)
    
    # Volume Regime Adjustment
    final_factor = np.where(data['high_volume_days'], final_factor * 1.2, final_factor)
    final_factor = np.where(data['low_volume_days'], final_factor * 0.8, final_factor)
    final_factor = np.where(data['volatility_regime_change'], final_factor * data['volatility_weight'], final_factor)
    
    # Return the factor series
    return pd.Series(final_factor, index=data.index, name='multi_timeframe_acceleration_divergence')
