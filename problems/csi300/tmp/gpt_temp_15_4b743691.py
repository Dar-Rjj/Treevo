import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Adaptive Acceleration-Memory Factor
    """
    data = df.copy()
    
    # Helper function for True Range
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    # Calculate log prices and volumes
    data['log_close'] = np.log(data['close'])
    data['log_volume'] = np.log(data['volume'])
    
    # Multi-Scale Acceleration Analysis
    # Price Acceleration Calculation
    # Short-term acceleration (5-day scale)
    data['price_accel_short'] = ((data['log_close'] - data['log_close'].shift(1)) - 
                                (data['log_close'].shift(1) - data['log_close'].shift(2))) - \
                               ((data['log_close'].shift(1) - data['log_close'].shift(2)) - 
                                (data['log_close'].shift(2) - data['log_close'].shift(3)))
    
    # Medium-term acceleration (10-day scale)
    data['price_accel_medium'] = ((data['log_close'] - data['log_close'].shift(5)) - 
                                 (data['log_close'].shift(5) - data['log_close'].shift(10))) - \
                                ((data['log_close'].shift(5) - data['log_close'].shift(10)) - 
                                 (data['log_close'].shift(10) - data['log_close'].shift(15)))
    
    # Long-term acceleration (20-day scale)
    data['price_accel_long'] = ((data['log_close'] - data['log_close'].shift(10)) - 
                               (data['log_close'].shift(10) - data['log_close'].shift(20))) - \
                              ((data['log_close'].shift(10) - data['log_close'].shift(20)) - 
                               (data['log_close'].shift(20) - data['log_close'].shift(30)))
    
    # Volume Acceleration Calculation
    # Short-term acceleration (5-day scale)
    data['volume_accel_short'] = ((data['log_volume'] - data['log_volume'].shift(1)) - 
                                 (data['log_volume'].shift(1) - data['log_volume'].shift(2))) - \
                                ((data['log_volume'].shift(1) - data['log_volume'].shift(2)) - 
                                 (data['log_volume'].shift(2) - data['log_volume'].shift(3)))
    
    # Medium-term acceleration (10-day scale)
    data['volume_accel_medium'] = ((data['log_volume'] - data['log_volume'].shift(5)) - 
                                  (data['log_volume'].shift(5) - data['log_volume'].shift(10))) - \
                                 ((data['log_volume'].shift(5) - data['log_volume'].shift(10)) - 
                                  (data['log_volume'].shift(10) - data['log_volume'].shift(15)))
    
    # Long-term acceleration (20-day scale)
    data['volume_accel_long'] = ((data['log_volume'] - data['log_volume'].shift(10)) - 
                                (data['log_volume'].shift(10) - data['log_volume'].shift(20))) - \
                               ((data['log_volume'].shift(10) - data['log_volume'].shift(20)) - 
                                (data['log_volume'].shift(20) - data['log_volume'].shift(30)))
    
    # Acceleration-Memory Integration
    # Historical acceleration pattern matching (20-day lookback)
    for col in ['price_accel_short', 'price_accel_medium', 'price_accel_long',
                'volume_accel_short', 'volume_accel_medium', 'volume_accel_long']:
        data[f'{col}_persistence'] = data[col].rolling(window=20, min_periods=10).apply(
            lambda x: np.corrcoef(x.iloc[-5:], np.arange(5))[0,1] if len(x.dropna()) >= 5 else 0, raw=False
        )
    
    # Volatility-Regime Adaptive Processing
    # True Range calculation
    data['true_range'] = true_range(data['high'], data['low'], data['close'].shift(1))
    
    # Multi-Timeframe Volatility State Assessment
    data['vol_ratio'] = (data['true_range'].rolling(window=5, min_periods=3).mean() / data['close']) / \
                       (data['true_range'].rolling(window=20, min_periods=10).mean() / data['close'])
    
    # Regime Classification
    data['vol_regime'] = np.where(data['vol_ratio'] > 1.2, 'high', 
                                 np.where(data['vol_ratio'] < 0.8, 'low', 'normal'))
    
    # Regime-Adaptive Acceleration Processing
    data['vol_adjusted_accel_short'] = np.where(
        data['vol_regime'] == 'high',
        data['price_accel_short'] * data['vol_ratio'] / (data['true_range'].rolling(window=5, min_periods=3).mean() / data['close']),
        data['price_accel_short']
    )
    
    data['vol_adjusted_accel_medium'] = np.where(
        data['vol_regime'] == 'high',
        data['price_accel_medium'] * data['vol_ratio'] / (data['true_range'].rolling(window=5, min_periods=3).mean() / data['close']),
        data['price_accel_medium']
    )
    
    # Microstructure Acceleration Enhancement
    # Spread acceleration
    data['range_ratio'] = (data['high'] - data['low']) / (data['close'] - data['open']).replace(0, np.nan)
    data['log_range_ratio'] = np.log(data['range_ratio'].replace(0, np.nan))
    data['spread_accel'] = ((data['log_range_ratio'] - data['log_range_ratio'].shift(1)) - 
                           (data['log_range_ratio'].shift(1) - data['log_range_ratio'].shift(2))) - \
                          ((data['log_range_ratio'].shift(1) - data['log_range_ratio'].shift(2)) - 
                           (data['log_range_ratio'].shift(2) - data['log_range_ratio'].shift(3)))
    
    # Intraday momentum acceleration
    data['intraday_return'] = data['close'] / data['open'] - 1
    data['intraday_accel'] = ((data['intraday_return'] - data['intraday_return'].shift(1)) - 
                             (data['intraday_return'].shift(1) - data['intraday_return'].shift(2))) - \
                            ((data['intraday_return'].shift(1) - data['intraday_return'].shift(2)) - 
                             (data['intraday_return'].shift(2) - data['intraday_return'].shift(3)))
    
    # Price-Memory Context Integration
    # Short-term reversal component
    data['short_reversal'] = ((data['close'] - data['low'].rolling(window=5, min_periods=3).min()) / data['close']) - \
                            ((data['high'].rolling(window=5, min_periods=3).max() - data['close']) / 
                             data['high'].rolling(window=5, min_periods=3).max())
    
    # Medium-term reversal component
    data['medium_reversal'] = ((data['close'] - data['low'].rolling(window=20, min_periods=10).min()) / data['close']) - \
                             ((data['high'].rolling(window=20, min_periods=10).max() - data['close']) / 
                              data['high'].rolling(window=20, min_periods=10).max())
    
    # Historical pattern matching for acceleration-memory consistency
    data['accel_consistency'] = (
        data['price_accel_short_persistence'] + 
        data['price_accel_medium_persistence'] + 
        data['volume_accel_short_persistence']
    ) / 3
    
    # Composite Factor Generation
    # Multi-Scale Acceleration-Memory Score
    data['accel_memory_score'] = (
        data['vol_adjusted_accel_short'] * 0.4 +
        data['vol_adjusted_accel_medium'] * 0.3 +
        data['price_accel_long'] * 0.3
    ) * data['accel_consistency']
    
    # Volatility-Adaptive Processing Score
    data['vol_adaptive_score'] = (
        data['vol_adjusted_accel_short'] * 0.3 +
        data['spread_accel'] * 0.25 +
        data['intraday_accel'] * 0.25 +
        data['volume_accel_short'] * 0.2
    ) * np.where(data['vol_regime'] == 'high', 1.2, 
                np.where(data['vol_regime'] == 'low', 0.8, 1.0))
    
    # Price-Memory Context Score
    data['price_memory_score'] = (
        data['short_reversal'] * 0.6 +
        data['medium_reversal'] * 0.4
    ) * data['accel_consistency']
    
    # Final Alpha Factor
    alpha_factor = (
        data['accel_memory_score'] * data['price_memory_score'] * 
        np.where(data['vol_regime'] == 'high', 1.1, 
                np.where(data['vol_regime'] == 'low', 0.9, 1.0))
    )
    
    return alpha_factor
